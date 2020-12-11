import tensorflow as tf
import pathlib
import numpy as np
import os

def get_label(file_path,class_names):
    
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names

    return tf.argmax(one_hot)

def decode_img(img,img_size):

    img = tf.image.decode_jpeg(img, channels=3)

    return tf.image.resize(img, [img_size, img_size])

def process_path(file_path,img_size,class_names):
    label = get_label(file_path,class_names)
    img = tf.io.read_file(file_path)
    img = decode_img(img,img_size)
    return img, label

def scale(image,label):
    image = tf.cast(image, tf.float32)
    image /= 255
    
    return image, label

def configure_for_performance(ds,batch_size,train=True):
    if train:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(scale,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds

def make_dataset_keras(train_path,val_path,batch_size,img_size):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_size,img_size),
        shuffle=True)

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_size,img_size),
        shuffle=False)

    train_dataset = train_dataset.map(scale,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(scale)
    
    return train_dataset,val_dataset

def make_dataset_tf(train_path,val_path,batch_size,img_size):
    train_data_dir = pathlib.Path(train_path)
    val_data_dir = pathlib.Path(val_path)

    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*') if item.name != "LICENSE.txt"]))
    class_names_val = np.array(sorted([item.name for item in val_data_dir.glob('*') if item.name != "LICENSE.txt"]))

    train_list_ds = tf.data.Dataset.list_files(f'{train_data_dir}/*/*.JPEG', shuffle=False)
    train_list_ds = train_list_ds.shuffle(len(train_list_ds), reshuffle_each_iteration=False)
    
    train_ds = train_list_ds.map(lambda x:process_path(x,img_size,class_names), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds.apply(tf.data.experimental.ignore_errors())
    train_ds = configure_for_performance(train_ds,batch_size)
    
    val_data_dir = pathlib.Path(val_path)

    val_list_ds = tf.data.Dataset.list_files(f'{val_data_dir}/*/*.JPEG', shuffle=False)
    
    val_ds = val_list_ds.map(lambda x:process_path(x,img_size,class_names_val), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = configure_for_performance(val_ds,batch_size,train=False)
    
    return train_ds, val_ds