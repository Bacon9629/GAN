import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.activations import sigmoid, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf


class Discriminator(models.Model):

    def __init__(self, summary=False, exist_weight=""):
        super().__init__(name="Discriminator")
        self.m = models.Sequential([
            Conv2D(64, (5, 5), strides=(2, 2), input_shape=[28, 28, 1]),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            BatchNormalization(),
            LeakyReLU(),
            Flatten(),
            Dense(1)
        ])

        if summary:
            self.m.summary()

        if exist_weight:
            self.load_weights(exist_weight)

    def call(self, inputs, training=None, mask=None):
        # [, 28, 28, 1]
        # num, img = tf.split(inputs, [1, 28], axis=1)
        result = self.m(inputs)  # [-1, 28, 28, 1] => [-1, 1]

        return result  # [-1, 1]

    def get_config(self):
        pass


class Classification(models.Model):
    """
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    """

    def __init__(self, summary=False, exist_weight=""):
        """

        :param exist_weight: 權重路降
        :param summary: 是否印出網路結構
        """
        super().__init__(name='Classification')
        self.m = models.Sequential([
                Conv2D(input_shape=(28, 28, 1), kernel_size=3, filters=16, strides=2),
                Flatten(),
                Dense(256, activation=relu),
                Dense(10, activation=softmax),
            ]
        )

        if summary:
            self.m.summary()
        if exist_weight:
            self.load_weights(exist_weight)

    def call(self, inputs, training=None, mask=None):
        # input: [28, 28, 1]
        # num_label, img_label = tf.split(inputs, [1, 28], axis=1)
        # num_label = tf.reshape(num_label, [-1, 28])

        return self.m(inputs)  # [, 10]

    def get_config(self):
        pass


class Generate(models.Model):

    def __init__(self, summary=False, exist_weight=""):
        super().__init__(name="Generate")
        self.m = models.Sequential([
            Dense(7 * 7 * 256, use_bias=False, input_shape=(128,)),
            BatchNormalization(),
            LeakyReLU(),
            Reshape((7, 7, 256)),
            Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),
            Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'),
        ])

        if summary:
            self.m.summary()
        if exist_weight:
            self.load_weights(exist_weight)


    def call(self, inputs, training=None, mask=None):
        #  inputs_shape: [, 128]  num_label_onehot: 28 + noise: 100
        num, noise = tf.split(inputs, [28, 100], axis=1)
        num = tf.reshape(num, [-1, 1, 28, 1])

        my_img = self.m(inputs)  # [-1, 28, 28, 1]
        result = tf.concat([num, my_img], axis=1)  # [-1, 29, 28, 1]
        # result = tf.reshape(result, [-1, 29, 28, 1])  # [-1, 29, 28, 1]

        return result


    def get_config(self):
        pass





if __name__ == '__main__':
    model = Discriminator()
    test = tf.random.normal([1, 28, 28, 1])
    result = model(test, training=False)
    print(result.shape)





