def main_gan_test():
    import layers
    from source import main_gan
    from keras.datasets.mnist import load_data
    from numpy import expand_dims
    import os
    
    os.makedirs('test_02', exist_ok=True)
    init = layers.initializer('RandomNormal', stddev=0.02)
    input_shape = (28, 28, 1)
    optimizer = layers.optimizer('Adam', learning_rate=0.0002, beta_1=0.5)
    latent_dim = 40
    
    """
    discriminator model
    Model: "sequential"
    __________________________________________________
    Layer (type)                 Output Shape         
    ==================================================
    conv2d (Conv2D)              (None, 14, 14, 64)   
    __________________________________________________
    batch_normalization (BatchNo (None, 14, 14, 64)   
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 14, 14, 64)   
    __________________________________________________
    conv2d (Conv2D)              (None, 7, 7, 64)     
    __________________________________________________
    batch_normalization (BatchNo (None, 7, 7, 64)     
    __________________________________________________
    leaky_re_lu (LeakyReLU)      (None, 7, 7, 64)     
    __________________________________________________
    flatten (Flatten)            (None, 3136)         
    __________________________________________________
    dense (Dense)                (None, 1)            
    ==================================================
    """
    argv_discriminator = {
        'model_type': 'Sequential',
        'lyrs': [
            # downsample to 14x14
            layers.layer('Conv2D', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init, input_shape=input_shape),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # downsample to 7x7
            layers.layer('Conv2D', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # classifier
            layers.layer('Flatten'),
            layers.layer('Dense', units=1, activation='linear', kernel_initializer=init)
        ],
        'optimizer': optimizer,
        'loss': 'mse'
    }
    
    """
    generator model
    Model: "sequential"
    __________________________________________________
    Layer (type)                 Output Shape            
    ==================================================
    dense (Dense)                (None, 6272)             
    __________________________________________________
    batch_normalization (BatchNo (None, 6272)              
    __________________________________________________
    activation (Activation)      (None, 6272)                  
    __________________________________________________
    reshape (Reshape)            (None, 7, 7, 128)             
    __________________________________________________
    conv2d_transpose (Conv2DTran (None, 14, 14, 128)      
    __________________________________________________
    batch_normalization (BatchNo (None, 14, 14, 128)         
    __________________________________________________
    activation (Activation)      (None, 14, 14, 128)           
    __________________________________________________
    conv2d_transpose (Conv2DTran (None, 28, 28, 64)       
    __________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 64)          
    __________________________________________________
    activation (Activation)      (None, 28, 28, 64)            
    __________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 1)          
    __________________________________________________
    activation (Activation)      (None, 28, 28, 1)             
    ==================================================
    """
    argv_generator = {
        'model_type': 'Sequential',
        'lyrs': [
            # foundation for 7x7 image
            layers.layer('Dense', units=128 * 7 * 7, kernel_initializer=init, input_dim=latent_dim),
            layers.layer('BatchNormalization'),
            layers.layer('Activation', activation='relu'),
            layers.layer('Reshape', target_shape=(7, 7, 128)),
            # upsample to 14x14
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('Activation', activation='relu'),
            # upsample to 28x28
            layers.layer('Conv2DTranspose', filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('Activation', activation='relu'),
            # output 28x28x1
            layers.layer('Conv2D', filters=1, kernel_size=(7, 7), padding='same', kernel_initializer=init),
            layers.layer('Activation', activation='tanh'),
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
        'loss': 'mse',
        'model_type': 'Sequential'
    }
    argv_train = {'train': train_15, 'latent_dim': latent_dim, 'do_print': True}

    # load data
    (train_x, _), (_, _) = load_data()
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
def train_15(g_model, d_model, gan_model, dataset, latent_dim,
             n_epochs=20, n_batch=64, do_print=False, save_model=True):
    from source import generate_real_samples, generate_fake_samples, generate_latent_points
    from numpy import ones
    # calculate the number of batches per training epoch
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # prepare real and fake samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator✬s error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        if do_print:
            print(f'>{i+1:03d}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}')
        # save the generator model
        if save_model:
            if (i+1)%batch_per_epoch == 0:
                g_model.save(f'test_02/generator_{(i+1)//batch_per_epoch:03d}.h5')


if __name__ == '__main__':
    main_gan_test()
