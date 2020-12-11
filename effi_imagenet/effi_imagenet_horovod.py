import tensorflow as tf
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import sys
from make_dataset import make_dataset_keras,make_dataset_tf
import horovod.tensorflow.keras as hvd
import argparse

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_SHM_DISABLE"] = "1"

parser = argparse.ArgumentParser(description='Train EfficientNetB0 on ImageNet Dataset',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-b', '--batch', type=int, default=64,
                    help='batch_size_per_replica')
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--random-seed', type=int, default=2386)
parser.add_argument('-e','--epochs', type=int, default=10)

parser.add_argument('--data-path', type=str, default="/image")
parser.add_argument('--checkpoint-dir',type=str, default='./training_checkpoints')
parser.add_argument('--logdir', type=str, default='./logs')
parser.add_argument('--verbose', type=int, default=1)

parser.add_argument('--no-prefetch', action='store_true',
                    help='disable dataset prefetching')
parser.add_argument('--no-tf-dataset', action='store_true',
                    help='disable tensorflow dataset')
parser.add_argument('--fp16', action='store_true',
                    help='enable fp16 mixed precision computing')

args = parser.parse_args()


DATA_PATH = args.data_path
TRAIN_PATH = os.path.join(DATA_PATH,"train")
VAL_PATH = os.path.join(DATA_PATH,"val")


hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

if args.fp16:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)


strategy = None
BATCH_SIZE = args.batch

if not args.no_tf_dataset:
    train_dataset, val_dataset = make_dataset_tf(TRAIN_PATH,VAL_PATH,
                                                    BATCH_SIZE,
                                                    args.image_size)
else:
    train_dataset, val_dataset = make_dataset_keras(TRAIN_PATH,VAL_PATH,
                                                    BATCH_SIZE,
                                                    args.image_size)

if not args.no_prefetch:
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
TRAIN_LENGTH = len(train_dataset)

optimizer = tf.keras.optimizers.Adam()
optimizer = hvd.DistributedOptimizer(optimizer)
model = tf.keras.applications.EfficientNetB0()
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=optimizer,
                metrics=['accuracy'],
                experimental_run_tf_function=False)


checkpoint_prefix = os.path.join(args.checkpoint_dir,"ckpt_{epoch}")

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

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=args.logdir,
                                    profile_batch = 0),
    tf.keras.callbacks.LearningRateScheduler(decay),
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    PrintLR()
]

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,save_weights_only=True))


model.fit(train_dataset, 
            validation_data=val_dataset,
            epochs=args.epochs, 
            callbacks=callbacks,
            steps_per_epoch = len(train_dataset)//hvd.size(),
            verbose = args.verbose if hvd.rank() == 0 else 0) #, validation_data=val_dataset )
