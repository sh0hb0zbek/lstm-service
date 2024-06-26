def main_gan_test():
    import layers
    from source import main_gan, ClipConstraint, wasserstein_loss
    from keras.datasets.mnist import load_data
    from numpy import expand_dims
    import os
    
    os.makedirs('test_03', exist_ok=True)
    input_shape = (28, 28, 1)
    optimizer = layers.optimizer('Adam', learning_rate=0.0002, beta_1=0.5)
    latent_dim = 100
    
    """
    discriminator model
    Model: "sequential"
    __________________________________________________
    Layer (type)                 Output Shape         
    ==================================================
    conv2d (Conv2D)              (None, 14, 14, 128)  
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 14, 14, 128)  
    __________________________________________________
    conv2d (Conv2D)              (None, 7, 7, 128)    
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 7, 7, 128)    
    __________________________________________________
    flatten (Flatten)            (None, 6272)         
    __________________________________________________
    dropout (Dropout)            (None, 6272)         
    __________________________________________________
    dense (Dense)                (None, 1)            
    ==================================================
    """
    argv_discriminator = {
        'model_type': 'Sequential',
        'lyrs': [
            # downsample to 14x14
            layers.layer('Conv2D', filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', input_shape=input_shape),
            layers.layer('LeakyReLU', alpha=0.2),
            # downsample to 7x7
            layers.layer('Conv2D', filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same'),
            layers.layer('LeakyReLU', alpha=0.2),
            # classifier
            layers.layer('Flatten'),
            layers.layer('Dropout', rate=0.4),
            layers.layer('Dense', units=1, activation='sigmoid')
        ],
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    
    """
    generator model
    Model: "sequential_37"
    __________________________________________________
    Layer (type)                 Output Shape         
    ==================================================
    dense (Dense)                (None, 6272)         
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 6272)         
    __________________________________________________
    reshape (Reshape)            (None, 7, 7, 128)    
    __________________________________________________
    conv2d_transpose (Conv2DTran (None, 14, 14, 128)  
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 14, 14, 128)  
    __________________________________________________
    conv2d_transpose (Conv2DTran (None, 28, 28, 64)   
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 28, 28, 64)   
    __________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 1)    
    ==================================================
    """
    argv_generator = {
        'model_type': 'Sequential',
        'lyrs': [
            # foundation for 7x7 image
            layers.layer('Dense', units=128 * 7 * 7, input_dim=latent_dim),
            layers.layer('LeakyReLU', alpha=0.2),
            layers.layer('Reshape', target_shape=(7, 7, 128)),
            # upsample to 14x14
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            layers.layer('LeakyReLU', alpha=0.2),
            # upsample to 28x28
            layers.layer('Conv2DTranspose', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same'),
            layers.layer('LeakyReLU', alpha=0.2),
            # output 28x28x1
            layers.layer('Conv2D', filters=1, kernel_size=(7, 7), activation='tanh', padding='same')
        ],
        'do_compile': False
    }
    
    """
    GAN model
    Model: "sequential"
    __________________________________________________
    Layer (type)                 Output Shape         
    ==================================================
    sequential (Sequential)      (None, 28, 28, 1)    
    __________________________________________________
    sequential (Sequential)      (None, 1)            
    ==================================================
    """
    argv_gan = {
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'model_type': 'Sequential'
    }
    argv_train = {'train': train_17, 'latent_dim': latent_dim, 'do_print': True}

    # load data
    (train_x, train_y), (_, _) = load_data()
    # expand to 3d
    x = expand_dims(train_x, axis=-1)
    # convert from ints to floats and scale from [0, 255] to [-1, 1]
    x = x.astype('float32')
    dataset = (x - 127.5) / 127.5

    argv = [
        argv_discriminator,
        argv_generator,
        argv_gan,
        argv_train,
        dataset
    ]
    
    main_gan(argv)


# train the generator and discriminator
def train_17(g_model, d_model, gan_model, dataset, latent_dim,
             n_epochs=100, n_batch=128, do_print=False, save_model=True):
    from source import generate_real_samples, generate_fake_samples, generate_latent_points
    from numpy import ones
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(batch_per_epoch):
            # get randomly selected ✬real✬ samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate ✬fake✬ examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator✬s error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            if do_print:
                print(f'>{i+1:03d}, {j+1:03d}/{batch_per_epoch}, d1={d_loss1:.3f}, d2={d_loss2:.3f} g={g_loss:.3f}')
        # save the generator model
        if save_model:
            g_model.save(f'test03/generator_{i+1:03d}.h5')


if __name__ == '__main__':
    main_gan_test()
