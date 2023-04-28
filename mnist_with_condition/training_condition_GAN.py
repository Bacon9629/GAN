import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.activations import sigmoid, softmax, relu
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
import tensorflow as tf
import tqdm

import MyModel as M


# Prepare the MNIST dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 512
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Initialize the generator and discriminator models
generator = M.Generate()
discriminator = M.Discriminator()
classification = M.Classification(exist_weight="./model_file/classification/classification.weight")

# Define the number of epochs for training
EPOCHS = 2000

# Define lists to store generator and discriminator losses for each epoch
gen_loss_history = []
disc_loss_history = []


cross_entropy = BinaryCrossentropy(from_logits=True)
m_cross = CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


# Define the training loop
def generator_loss(fake_output, img_label, img_classify_predict):
    # a = cross_entropy(tf.ones_like(fake_output), fake_output)
    # b = binary_cross_entropy(img_label, img_classify_predict)
    return cross_entropy(tf.ones_like(fake_output), fake_output) + m_cross(img_label, img_classify_predict)
    # return a + b

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def get_one_label_noise(BATCH_SIZE):
    result = tf.random.normal(shape=[BATCH_SIZE, 100])
    b = np.random.randint(0, 10, size=[BATCH_SIZE, 1])
    b = tf.one_hot(b, 28)
    b = tf.reshape(b, [BATCH_SIZE, 28])

    result = tf.concat([b, result], axis=1)
    # print(result.shape)
    return result


@tf.function
def train_step(images):
    # noise = tf.random.normal([BATCH_SIZE, 100])
    label_noise = get_one_label_noise(BATCH_SIZE)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(label_noise, training=True)  # [-1, 128] => [-1, 29, 28, 1]

        m_num, m_img = tf.split(generated_images, [1, 28], axis=1)  # [-1, 28, 1, 1], [-1, 28, 28, 1]
        m_num = m_num[:, 0, :10, 0]

        m_img_predict_num_result = classification(m_img, training=False)  # [-1, 10]

        fake_output = discriminator(m_img, training=True)  # # [-1, 1]
        real_output = discriminator(images, training=True)  # [-1, 1]
        # print(m_num.shape)
        gen_loss = generator_loss(fake_output, m_num, m_img_predict_num_result)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


# Start training loop
for epoch in range(EPOCHS):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    for images in tqdm.tqdm(train_dataset):
        gen_loss, disc_loss = train_step(images)
    if epoch % 20:
        # Save the generator model
        generator.save_weights('./model_file/generator/generator_model.weight')
        discriminator.save_weights('./model_file/discriminator/discriminator_model.weight')
    if epoch == 500:
        generator.save_weights('./model_file/generator/generator_model_500.weight')
        discriminator.save_weights('./model_file/discriminator/discriminator_model_500.weight')
    if epoch == 1000:
        generator.save_weights('./model_file/generator/generator_model_1000.weight')
        discriminator.save_weights('./model_file/discriminator/discriminator_model_1000.weight')
    if epoch == 1500:
        generator.save_weights('./model_file/generator/generator_model_1500.weight')
        discriminator.save_weights('./model_file/discriminator/discriminator_model_1500.weight')


    # Save generator and discriminator losses for this epoch
    # gen_loss_history.append(gen_loss)
    # disc_loss_history.append(disc_loss)

    # Print generator and discriminator losses for this epoch
    print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}'.format(epoch+1, gen_loss, disc_loss))

# Save the generator model
generator.save_weights('./model_file/generator/generator_model.weight')
discriminator.save_weights('./model_file/discriminator/discriminator_model.weight')
