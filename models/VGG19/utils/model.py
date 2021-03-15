import os
from io import BytesIO
import time
import requests
import base64
from pathlib import Path
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import load_model,Model
import matplotlib.pyplot as plt
import matplotlib

# Loss model to calculate perceptual loss

class LossModel:
    def __init__(self, pretrained_model, content_layers, style_layers):
        self.model = pretrained_model
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.loss_model = self.get_model()

    def get_model(self):
        self.model.trainable = False
        layer_names = self.style_layers + self.content_layers
        outputs = [self.model.get_layer(name).output for name in layer_names]
        new_model = Model(inputs=self.model.input, outputs=outputs)
        return new_model

    def get_activations(self, inputs):
        inputs = inputs*255.0
        style_length = len(self.style_layers)
        outputs = self.loss_model(vgg19.preprocess_input(inputs))
        style_output, content_output = outputs[:
                                               style_length], outputs[style_length:]
        content_dict = {name: value for name, value in zip(
            self.content_layers, content_output)}
        style_dict = {name: value for name, value in zip(
            self.style_layers, style_output)}
        return {'content': content_dict, 'style': style_dict}


# Model layers

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'REFLECT')


class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        batch, rows, cols, channels = [i for i in inputs.get_shape()]
        mu, var = tf.nn.moments(inputs, [1, 2], keepdims=True)
        shift = tf.Variable(tf.zeros([channels]))
        scale = tf.Variable(tf.ones([channels]))
        scale = tf.cast(scale, tf.float32)
        epsilon = 1e-3
        normalized = (inputs-mu)/tf.sqrt(var + epsilon)
        normalized = tf.cast(normalized, tf.float32)
        shift = tf.cast(shift, tf.float32)
        return scale * normalized + shift


class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        self.padding = ReflectionPadding2D([k//2 for k in kernel_size])
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides)
        self.bn = InstanceNormalization()

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.conv2d(x)
        x = self.bn(x)
        return x


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(ResidualLayer, self).__init__(**kwargs)
        self.conv2d_1 = ConvLayer(filters, kernel_size)
        self.conv2d_2 = ConvLayer(filters, kernel_size)
        self.relu = tf.keras.layers.ReLU()
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        residual = inputs
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.add([x, residual])
        return x


class UpsampleLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, upsample=2, **kwargs):
        super(UpsampleLayer, self).__init__(**kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=upsample)
        self.padding = ReflectionPadding2D([k//2 for k in kernel_size])
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel_size, strides)
        self.bn = InstanceNormalization()

    def call(self, inputs):
        x = self.upsample(inputs)
        x = self.padding(x)
        x = self.conv2d(x)
        return self.bn(x)

# Style transfer model


class StyleTransferModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(StyleTransferModel, self).__init__(
            name='StyleTransferModel', **kwargs)
        self.conv2d_1 = ConvLayer(filters=32, kernel_size=(
            9, 9), strides=1, name="conv2d_1_32")
        self.conv2d_2 = ConvLayer(filters=64, kernel_size=(
            3, 3), strides=2, name="conv2d_2_64")
        self.conv2d_3 = ConvLayer(filters=128, kernel_size=(
            3, 3), strides=2, name="conv2d_3_128")
        self.res_1 = ResidualLayer(
            filters=128, kernel_size=(3, 3), name="res_1_128")
        self.res_2 = ResidualLayer(
            filters=128, kernel_size=(3, 3), name="res_2_128")
        self.res_3 = ResidualLayer(
            filters=128, kernel_size=(3, 3), name="res_3_128")
        self.res_4 = ResidualLayer(
            filters=128, kernel_size=(3, 3), name="res_4_128")
        self.res_5 = ResidualLayer(
            filters=128, kernel_size=(3, 3), name="res_5_128")
        self.deconv2d_1 = UpsampleLayer(
            filters=64, kernel_size=(3, 3), name="deconv2d_1_64")
        self.deconv2d_2 = UpsampleLayer(
            filters=32, kernel_size=(3, 3), name="deconv2d_2_32")
        self.deconv2d_3 = ConvLayer(filters=3, kernel_size=(
            9, 9), strides=1, name="deconv2d_3_3")
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.conv2d_3(x)
        x = self.relu(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)
        x = self.deconv2d_1(x)
        x = self.relu(x)
        x = self.deconv2d_2(x)
        x = self.relu(x)
        x = self.deconv2d_3(x)
        x = (tf.nn.tanh(x) + 1) * (255.0 / 2)
        return x

    # Used to print shapes of each layer to check if input shape == output shape
    def print_shape(self, inputs):
        print(inputs.shape)
        x = self.conv2d_1(inputs)
        print(x.shape)
        x = self.relu(x)
        x = self.conv2d_2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.conv2d_3(x)
        print(x.shape)
        x = self.relu(x)
        x = self.res_1(x)
        print(x.shape)
        x = self.res_2(x)
        print(x.shape)
        x = self.res_3(x)
        print(x.shape)
        x = self.res_4(x)
        print(x.shape)
        x = self.res_5(x)
        print(x.shape)
        x = self.deconv2d_1(x)
        print(x.shape)
        x = self.relu(x)
        x = self.deconv2d_2(x)
        print(x.shape)
        x = self.relu(x)
        x = self.deconv2d_3(x)
        print(x.shape)
