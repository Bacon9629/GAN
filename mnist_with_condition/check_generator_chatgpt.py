import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the generator model
generator = tf.keras.models.load_model('generator_model.h5')

# Generate 25 random noise vectors
noise = tf.random.normal([25, 100])

# Use the generator to generate images from the noise
generated_images = generator(noise, training=False)

# Rescale the generated images from [-1, 1] to [0, 1] for display
generated_images = (generated_images + 1) / 2

# Plot the generated images
fig, axs = plt.subplots(5, 5)
count = 0
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        count += 1
plt.show()
