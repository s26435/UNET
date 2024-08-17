import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from hyper_parameters import img_size, img_channels
import numpy as np


class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, discriminator, timesteps, gdf_util, trainable=True, ema=0.999, disc_factor=0.1):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.discriminator = discriminator
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.trainable = trainable
        self.disc_factor = disc_factor 

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(
            minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64
        )

        # Generate noisy images
        noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
        images_t = self.gdf_util.q_sample(images, t, noise)

        with tf.GradientTape() as tape:
            # Predict noise using the network
            pred_noise = self.network([images_t, t], training=True)
            
            # Calculate the diffusion loss
            diffusion_loss = self.loss_fn(noise, pred_noise)

            # Generate fake images (sampled)
            generated_images = self.gdf_util.p_sample(pred_noise, images_t, t, clip_denoised=True)

            # Pass real and fake images through the discriminator
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate the adversarial loss
            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
            adversarial_loss = real_loss + fake_loss

            # Combine the diffusion loss with the adversarial loss
            total_loss = diffusion_loss + self.disc_factor * adversarial_loss

        # Calculate gradients and update the network weights
        gradients = tape.gradient(total_loss, self.network.trainable_weights)
        self.g_optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        # Update discriminator separately
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
            fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss

        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_weights))

        # Update EMA weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {"total_loss": total_loss, "diffusion_loss": diffusion_loss, "disc_loss": disc_loss}


    def generate_images(self, num_images=16):
        assert self.ema_network is not None, "ema_network is not initialized"
        samples = tf.random.normal(shape=(num_images, 2 * img_size, img_size, img_channels), dtype=tf.float32)
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            assert pred_noise is not None, "pred_noise is None"
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
        return samples

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)):
        if epoch % 500 == 0:
            generated_samples = self.generate_images(num_images=num_rows * num_cols)
            generated_samples = (
                tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
                .numpy()
                .astype(np.uint8)
            )
    
            _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
            for i, image in enumerate(generated_samples):
                if num_rows == 1:
                    ax[i].imshow(image)
                    ax[i].axis("off")
                else:
                    ax[i // num_cols, i % num_cols].imshow(image)
                    ax[i // num_cols, i % num_cols].axis("off")
    
            plt.tight_layout()
            plt.show()

def save_model(model, path):
    model.save_weights(path)

def load_model(model, path):
    model.load_weights(path)





"""
class DiffusionModel(keras.Model):
    def __init__(self, network, ema_network, timesteps, gdf_util, trainable=True, ema=0.999):
        super().__init__()
        self.network = network
        self.ema_network = ema_network
        self.timesteps = timesteps
        self.gdf_util = gdf_util
        self.ema = ema
        self.trainable = trainable

    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        t = tf.random.uniform(minval=0, maxval=self.timesteps, shape=(batch_size,), dtype=tf.int64)

        with tf.GradientTape() as tape:
            noise = tf.random.normal(shape=tf.shape(images), dtype=images.dtype)
            images_t = self.gdf_util.q_sample(images, t, noise)
            pred_noise = self.network([images_t, t], training=True)
            loss = self.loss(noise, pred_noise)

        gradients = tape.gradient(loss, self.network.trainable_weights)

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {"loss": loss}

    def generate_images(self, num_images=16):
        samples = tf.random.normal(shape=(num_images, 2 * img_size, img_size, img_channels), dtype=tf.float32)
        for t in reversed(range(0, self.timesteps)):
            tt = tf.cast(tf.fill(num_images, t), dtype=tf.int64)
            pred_noise = self.ema_network.predict(
                [samples, tt], verbose=0, batch_size=num_images
            )
            samples = self.gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
        return samples

    def plot_images(self, epoch=None, logs=None, num_rows=2, num_cols=8, figsize=(12, 5)):
        if epoch % 500 == 0:
            generated_samples = self.generate_images(num_images=num_rows * num_cols)
            generated_samples = (
                tf.clip_by_value(generated_samples * 127.5 + 127.5, 0.0, 255.0)
                .numpy()
                .astype(np.uint8)
            )
    
            _, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
            for i, image in enumerate(generated_samples):
                if num_rows == 1:
                    ax[i].imshow(image)
                    ax[i].axis("off")
                else:
                    ax[i // num_cols, i % num_cols].imshow(image)
                    ax[i // num_cols, i % num_cols].axis("off")
    
            plt.tight_layout()
            plt.show()



"""