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
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import matplotlib


class TensorflowDatasetLoader:
    def __init__(self, dataset_path, batch_size=4, image_size=(256, 256), num_images=None):
        images_paths = [str(path) for path in Path(dataset_path).glob("*.jpg")]
        self.length = len(images_paths)
        if num_images is not None:
            images_paths = images_paths[0:num_images]
        dataset = tf.data.Dataset.from_tensor_slices(images_paths).map(
            lambda path: self.load_tf_image(path, dim=image_size),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.dataset = dataset

    def __len__(self):
        return self.length

    def load_tf_image(self, image_path, dim):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, dim)
        image = image/255.0
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image
