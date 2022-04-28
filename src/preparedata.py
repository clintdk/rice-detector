import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf


class preparedata:
    def __init__(self, path: str, img_height=28, img_width=28, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.path = path
        self.__loaddata()
        # self.__normalizedata()

    def __loaddata(self):

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255, validation_split=0.2
        )

        self.train_ds = datagen.flow_from_directory(
            self.path,
            target_size=(self.img_width, self.img_height),
            color_mode="rgb",
            batch_size=self.batch_size,
            subset="training",
            seed=123,
        )

        self.val_ds = datagen.flow_from_directory(
            self.path,
            target_size=(self.img_width, self.img_height),
            color_mode="rgb",
            batch_size=self.batch_size,
            subset="validation",
            seed=123,
        )


class classifier(preparedata):
    def __init__(self, path: str, img_height=28, img_width=28, batch_size=32):
        super().__init__(path, img_height, img_width, batch_size)
        self.__model()
        self.__train()

    def __model(self):
        num_classes = 5

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(self.img_height, self.img_width, 3),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(axis=3),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(axis=3),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.BatchNormalization(axis=3),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def __train(self):
        self.history = self.model.fit_generator(
            self.train_ds, validation_data=self.val_ds, epochs=1
        )

    def trained_model(self):
        return self.history

    def save_model(self, savedir: str):
        self.history.save(savedir, save_format="tf")
