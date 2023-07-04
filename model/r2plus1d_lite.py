# Ref. https://www.tensorflow.org/tutorials/video/video_classification
## layers.Resizing->layers.Conv3D

import tensorflow as tf


def prepare(frame_num=16, height=192, width=192, classes_num=101):
    input_shape = (None, frame_num, height, width, 3)
    input = tf.keras.layers.Input(shape=(input_shape[1:]))

    # Block 0
    ## Conv2Plus1D
    x0 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 7, 7), padding='same')(input)
    x0 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 1, 1), padding='same')(x0)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x0 = tf.keras.layers.ReLU()(x0)
    ## Resize
    x0 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x0)

    # Block 1
    ## ResConv2Plus1D
    x1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 7, 7), padding='same')(x0)
    x1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 1, 1), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)
    x1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 7, 7), padding='same')(x1)
    x1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(3, 1, 1), padding='same')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Add()([x0, x1])
    ## Resize
    x1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x1)

    # Block 2
    ## ResConv2Plus1D
    x2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 7, 7), padding='same')(x1)
    x2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 1, 1), padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)
    x2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 7, 7), padding='same')(x2)
    x2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(3, 1, 1), padding='same')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x11 = tf.keras.layers.Dense(units=32)(x1)
    x11 = tf.keras.layers.BatchNormalization()(x11)
    x2 = tf.keras.layers.Add()([x11, x2])
    ## Resize
    x2 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 3, 3), strides=(1, 2, 2), padding='same')(x2)

    # Block 4
    ## ResConv2Plus1D
    x4 = tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 7, 7), padding='same')(x2)
    x4 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)
    x4 = tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 7, 7), padding='same')(x4)
    x4 = tf.keras.layers.Conv3D(filters=128, kernel_size=(3, 1, 1), padding='same')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x41 = tf.keras.layers.Dense(units=128)(x4)
    x41 = tf.keras.layers.BatchNormalization()(x41)
    x4 = tf.keras.layers.Add()([x41, x4])

    xo = tf.keras.layers.GlobalAveragePooling3D()(x4)
    xo = tf.keras.layers.Flatten()(xo)
    xo = tf.keras.layers.Dense(classes_num)(xo)

    model = tf.keras.Model(input, xo)
    return model