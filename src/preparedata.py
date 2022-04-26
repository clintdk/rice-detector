import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf


class preparedata:
    def __init__(self, path: str, img_height, img_width, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.path = path
        self.__loaddata()
        self.__normalizedata()
        self.__resizedata()
        self.__splitdata()

    def __loaddata(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
        )

    def __normalizedata(self):
        self.train_ds = self.train_ds / 255.0
        self.val_ds = self.val_ds / 255.0
