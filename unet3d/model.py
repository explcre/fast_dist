import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv3D, Input, MaxPooling3D, Dropout, concatenate, UpSampling3D, Lambda, Conv3DTranspose
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def Unet3D(inputs,num_classes):
    x=inputs
    conv1 = Conv3D(32, 3, activation = 'relu', padding = 'same')(x)
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)

    up6 = Conv3DTranspose(512, 2, activation = 'relu', strides=(2,2,2),padding = 'valid')(drop4)
    merge6 = concatenate([conv3,up6],axis=-1)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(256, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3DTranspose(256, 2, activation = 'relu', strides=(2,2,2),padding = 'valid')(conv6)
    merge7 = concatenate([conv2,up7],axis=-1)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3DTranspose(128, 2, activation = 'relu', strides=(2,2,2), padding = 'same')(conv7)
    merge8 = concatenate([conv1,up8],axis=-1)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv8)

    conv10 = Conv3D(num_classes, 1)(conv8)
    outputs = tf.keras.layers.Activation('sigmoid', dtype='float32')(conv10)

    model = Model(inputs=inputs, outputs = outputs)

    return model