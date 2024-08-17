import os
import tensorflow as tf
from tensorflow import keras
from gaussian_diffusion import GaussianDiffusion
from layers import build_model
from hyper_parameters import img_size, img_channels, widths, has_attention, num_res_blocks, norm_groups, total_timesteps, learning_rate, BATCHSIZE, EPOCHS, DATAPATH, IMAGE_SIZE
from diffusion_model import DiffusionModel, save_model
from image_processing import load_images_from_folder
from discriminator import build_discriminator

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )

    ema_network = build_model(
        img_size=img_size,
        img_channels=img_channels,
        widths=widths,
        has_attention=has_attention,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        activation_fn=keras.activations.swish,
    )

    ema_network.set_weights(network.get_weights())
    
    gdf_util = GaussianDiffusion(timesteps=total_timesteps)

    discriminator = build_discriminator((IMAGE_SIZE[0], IMAGE_SIZE[1], img_channels))

    model = DiffusionModel(
        network=network,
        discriminator= discriminator,
        ema_network=ema_network,
        gdf_util=gdf_util,
        timesteps=total_timesteps,
    )

    model.compile(
        loss_fn=keras.losses.MeanSquaredError(),
        g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )

    if os.path.exists('anime.weights.h5') and False:
        model.load_weights('anime.weights.h5')

    network.summary()

    x_train = load_images_from_folder(DATAPATH, IMAGE_SIZE)
    print(f"Loaded {len(x_train)} images of size {x_train.shape}.")


    model.fit(
    x_train,
    epochs=EPOCHS,
    batch_size=BATCHSIZE,
    callbacks=[keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images)],)

    model.build(((None, 2 * img_size, img_size, img_channels)))

    save_model(model, 'anime.weights.h5')
    
    gaussian_diffusion = GaussianDiffusion()

    x_start = x_train[0:1]

    timesteps_to_plot = tf.cast(tf.linspace(0, gaussian_diffusion.timesteps - 1, 10), tf.int32)

    plt.figure(figsize=(20, 2))  

    for i, t in enumerate(timesteps_to_plot):
        noise = tf.random.normal(shape=tf.shape(x_start))  
        x_t = gaussian_diffusion.q_sample(x_start, t, noise)  
        x_t = tf.clip_by_value(x_t, gaussian_diffusion.clip_min, gaussian_diffusion.clip_max)  
        plt.subplot(1, 10, i + 1)  
        plt.imshow((((x_t[0] + 1) / 2.0) * 255.0).numpy().astype("uint8"))  
        plt.axis("off")  
        plt.title(f't={t.numpy()}') 

    plt.show()
    gdf_util2 = GaussianDiffusion(timesteps=total_timesteps)
    model2 = DiffusionModel(
        network=network,
        discriminator=discriminator,
        ema_network=ema_network,
        gdf_util=gdf_util2,
        timesteps=total_timesteps,
    )
    if True:
        model2.load_weights('anime.weights.h5')
        num_rows = 5
        num_cols = 2
        figsize = (10, 10)
        generated_samples = model2.generate_images(num_images=num_rows * num_cols)
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