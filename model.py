#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Add, GlobalAveragePooling2D, DepthwiseConv2D, BatchNormalization, LeakyReLU
from keras.layers import Flatten, Dense, Dropout, Input
from keras.models import Sequential, Model

import argparse

MODELS = {
    "vgg16": VGG16,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
    "mobilenet": MobileNet,
    "custom":""
}

# Define path to pre-trained classification block weights - this is
vgg_weights_path = "weights/snapshot_vgg_weights.hdf5"
res_weights_path = "weights/snapshot_res_weights.hdf5"
mob_weights_path = "weights/snapshot_mob_weights.hdf5"
custom_model_path = "weights/keras.model1"





def create_model(model, model_weights_path=None, top_model=True, color_mode="rgb", input_shape=None):
    if model=="custom":
        inputs = Input(shape=(64, 64, 3))
        net = Conv2D(32, kernel_size=3, strides=1, padding="same")(inputs)
        net = LeakyReLU()(net)
        net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
        net = LeakyReLU()(net)
        net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
        net = LeakyReLU()(net)

        net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
        net = LeakyReLU()(net)
        net = Conv2D(32, kernel_size=3, strides=1, padding="same")(net)
        net = LeakyReLU()(net)
        net = Conv2D(32, kernel_size=3, strides=2, padding="same")(net)
        net = LeakyReLU()(net)

        shortcut = net

        net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
        net = BatchNormalization(axis=3)(net)
        net = LeakyReLU()(net)
        net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
        net = BatchNormalization(axis=3)(net)
        net = LeakyReLU()(net)

        net = DepthwiseConv2D(kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(net)
        net = BatchNormalization(axis=3)(net)
        net = LeakyReLU()(net)
        net = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(net)
        net = BatchNormalization(axis=3)(net)
        net = LeakyReLU()(net)

        net = Add()([net, shortcut])

        net = GlobalAveragePooling2D()(net)
        net = Dropout(0.2)(net)

        net = Dense(128, activation='relu')(net)
        outputs = Dense(29, activation='softmax')(net)

        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights('weights/keras.model1')
        return model

        
    if model not in MODELS.keys():
        raise AssertionError("The model parameter must be a key in the `MODELS` dictionary")

    if color_mode == "grayscale":
        num_channels = 1
    else:
        num_channels = 3
    print("[INFO] loading %s..." % (model,))
    model = MODELS[model](include_top=False,
                          input_shape=(224, 224, 3))

    if top_model:
        top_model = Sequential()
        top_model.add(Flatten(input_shape=model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dense(26, activation='softmax'))

        print("[INFO] loading model weights.")
        if model_weights_path is not None:
            top_model.load_weights(model_weights_path)
        elif model == "vgg16":
            top_model.load_weights(vgg_weights_path)
        elif model == "resnet":
            print("ResNet50 pre-trained weights are not available yet, please use VGG16 for now!")
            top_model.load_weights(res_weights_path)
        elif model == "mobnet":
            print("ResNet50 pre-trained weights are not available yet, please use VGG16 for now!")
            top_model.load_weights(mob_weights_path)

        print("[INFO] creating model.")
        my_model = Model(inputs=model.input,
                         outputs=top_model(model.output))
        return my_model
    else:
        return model
