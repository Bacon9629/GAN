import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.activations import sigmoid, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
import tensorflow as tf

import MyModel as M



# Prepare the MNIST dataset
(train_images, train_labels), (eva_img, eva_label) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
eva_img = eva_img.reshape(eva_img.shape[0], 28, 28, 1).astype('float32')
eva_img = (eva_img - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 128
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

classify = M.Classification()
classify.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classify.build(input_shape=[3, 28, 28, 1])
# classify.summary()

classify.fit(train_images, train_labels, epochs=5, batch_size=BATCH_SIZE)
classify.save_weights("./model_file/classification/classification.weight")
# classify.save("./model_file/classification.h5")



