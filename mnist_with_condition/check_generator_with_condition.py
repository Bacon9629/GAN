from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.activations import sigmoid, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
import tensorflow as tf

import MyModel as M
import numpy as np
import matplotlib.pyplot as plt

# Load the generator model
generator = M.Generate(exist_weight="./model_file/generator/generator_model_500.weight")
# generator.set_weights("./model_file/generator/generator_model.weight")

# Generate 25 random noise vectors
noise = tf.random.normal([30, 100])
b = np.arange(0, 10).repeat(3)

b = tf.one_hot(b, 28)
b = tf.reshape(b, [30, 28])
noise = tf.concat([b, noise], axis=1)


# Use the generator to generate images from the noise
generated_images = generator(noise, training=False)

# Rescale the generated images from [-1, 1] to [0, 1] for display
generated_images = (generated_images + 1) / 2

# Plot the generated images
fig, axs = plt.subplots(6, 5)
count = 0
for i in range(6):
    for j in range(5):
        axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        count += 1
plt.show()
