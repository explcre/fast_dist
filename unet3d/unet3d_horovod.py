import os
import sys
import argparse

import tensorflow as tf
import numpy as np

from make_dataset import *
from model import Unet3D
from utils import dice_coe, dice_loss

import horovod.tensorflow.keras as hvd

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SHM_DISABLE"] = "1"

parser = argparse.ArgumentParser(description='Train 3D-Unet on 3D image Dataset',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-b', '--batch', type=int, default=1,
                    help='batch_size_per_replica')

parser.add_argument('--patch-size', type=int, default=224)
parser.add_argument('--overlap-size', type=int, default=112)
parser.add_argument('--input-size', type=int, default=112)

parser.add_argument('--random-seed', type=int, default=2386)
parser.add_argument('-e','--epochs', type=int, default=10)

parser.add_argument('--data-path', type=str, default=None)
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--verbose', type=int, default=1)

parser.add_argument('--fp16', action='store_true',
                    help='enable fp16 mixed precision computing')

parser.add_argument('--jpeg-dataset', action='store_true',
                    help='enable fp16 mixed precision computing')
parser.add_argument('--patch-dataset', action='store_true',
                    help='enable fp16 mixed precision computing')
parser.add_argument('--patch-resized-dataset', action='store_true',
                    help='enable fp16 mixed precision computing')

args = parser.parse_args()

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)


RANDOM_SEED = args.random_seed
IMG_CHANNEL = 1
PATCH_SHAPE = (args.patch_size,args.patch_size,args.patch_size)
OVERLAP = (args.overlap_size,args.overlap_size,args.overlap_size)
EPOCHS = args.epochs

MIXED_PRECISION = args.fp16

INSTANT_JPEG_DATASET = args.jpeg_dataset
PATCHES_RESIZE_DATASET = args.patch_dataset
PATCHES_PRE_RESIZE_DATASET = args.patch_resized_dataset

if not INSTANT_JPEG_DATASET and not PATCHES_RESIZE_DATASET and not PATCHES_PRE_RESIZE_DATASET:
    PATCHES_PRE_RESIZE_DATASET = True

BATCH_SIZE = args.batch

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

if INSTANT_JPEG_DATASET:
    if args.data_path != None:
        data_path = args.data_path
    else:
        data_path = "dataset/kitech"
    input_path = os.path.join(data_path,"p2","front")
    target_path = os.path.join(data_path,"p2","front","label")


    dataset = InstantJpegDataset(BATCH_SIZE,input_path,target_path,\
                                PATCH_SHAPE,OVERLAP,shuffle=True)

elif PATCHES_RESIZE_DATASET:

    if args.data_path != None:
        data_path = args.data_path
    else:
        data_path = "dataset/kitech_patches"
    train_path = os.path.join(data_path,"train")
    test_path = os.path.join(data_path,"test")

    input_img_paths = []
    input_target_paths = []

    for img_name in os.listdir(train_path):
        if "_label_" not in img_name:
            img_path = os.path.join(train_path,img_name)
            img_target_path = img_path.split('_')
            img_target_path.insert(-1,"label")
            img_target_path = "_".join(img_target_path)
            input_img_paths.append(img_path)
            input_target_paths.append(img_target_path)

    dataset = PatchesResizeDataset(BATCH_SIZE,input_img_paths,input_target_paths)

elif PATCHES_PRE_RESIZE_DATASET:
    if args.data_path != None:
        data_path = args.data_path
    else:
        data_path = "dataset/kitech_patches_resized"
    train_path = os.path.join(data_path,"train")
    test_path = os.path.join(data_path,"test")

    input_img_pre_paths = []
    input_target_pre_paths = []

    for img_name in os.listdir(train_path):
        if "_label_" not in img_name:
            img_path = os.path.join(train_path,img_name)
            img_target_path = img_path.split('_')
            img_target_path.insert(-1,"label")
            img_target_path = "_".join(img_target_path)
            input_img_pre_paths.append(img_path)
            input_target_pre_paths.append(img_target_path)

    dataset = PatchesDataset(BATCH_SIZE,input_img_pre_paths,input_target_pre_paths)

model = Unet3D(tf.keras.Input(shape=(IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH,1)),1)

eps_value = 1e-08
if MIXED_PRECISION:
    eps_value = 1e-04

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    epsilon=eps_value,
)

optimizer = hvd.DistributedOptimizer(optimizer)

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

model.compile(loss=dice_loss,#loss_object,
            optimizer=optimizer,
            metrics=['accuracy',dice_coe],
            experimental_run_tf_function=False)

def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nLearning rate for epoch {epoch+1} is {model.optimizer.lr.numpy()}')

if hvd.rank()==0:
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                        histogram_freq = 1,
                                        profile_batch = 0),
        PrintLR(),
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
    verbose = args.verbose
else:
    callbacks=[hvd.callbacks.BroadcastGlobalVariablesCallback(0)]
    verbose = 0

model.fit(dataset,
            steps_per_epoch=len(dataset)//hvd.size(),
            epochs=EPOCHS, 
            callbacks=callbacks, 
            verbose= verbose)