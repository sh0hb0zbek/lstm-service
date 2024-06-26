def main_gan_test():
    import layers
    from source import main_gan
    from keras.datasets.mnist import load_data
    from numpy import expand_dims

    init = layers.initializer('RandomNormal', stddev=0.02)
    input_shape = (28, 28, 1)
    optimizer = layers.optimizer('Adam', learning_rate=0.0002, beta_1=0.5)
    n_nodes = 128 * 7 * 7
    latent_dim = 50

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
            layers.layer('Dense', units=1, activation='sigmoid')
        ],
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'metrics': ['accuracy']
    }
    argv_generator = {
        'model_type': 'Sequential',
        'lyrs': [
            # foundation for 7x7 image
            layers.layer('Dense', units=n_nodes, kernel_initializer=init, input_dim=latent_dim),
            layers.layer('LeakyReLU', alpha=0.2),
            layers.layer('Reshape', target_shape=(7, 7, 128)),
            # upsample to 14x14
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # upsample to 28x28
            layers.layer('Conv2DTranspose', filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=init),
            layers.layer('BatchNormalization'),
            layers.layer('LeakyReLU', alpha=0.2),
            # output 28x28x1
            layers.layer('Conv2D', filters=1, kernel_size=(7, 7), activation='tanh', padding='same', kernel_initializer=init)
        ],
        'do_compile': False
    }
    argv_gan = {
        'optimizer': optimizer,
        'loss': 'binary_crossentropy',
        'model_type': 'Sequential'
    }
    argv_train = {'train': train_10, 'latent_dim': latent_dim}

    # load data
    (train_x, train_y), (_, _) = load_data()
    # expand to 3d
    x = expand_dims(train_x, axis=-1)
    selected_ix = train_y == 8
    x = x[selected_ix]
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
def train_10(g_model, d_model, gan_model, dataset, latent_dim,
          n_epochs=10, n_batch=128, do_print=False, save_model=True):
    # calculate the number of batches per epoch
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the total iterations based on batch and epoch
    n_steps = bat_per_epo * n_epochs
    # calculate the number of samples in half a batch
    half_batch = int(n_batch / 2)
    # prepare lists for storing stats each iteration
    d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected ✬real✬ samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, d_acc1 = d_model.train_on_batch(X_real, y_real)
        # generate ✬fake✬ examples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, d_acc2 = d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator✬s error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        if do_print:
            print(f'>{i+1}, d1={d_loss1:.3f}, d2={d_loss2:.3f}, g={g_loss:.3f}, a1={100*d_acc1:.0f}, a2={100*d_acc2}')
        if save_model:
            if (i+1)%batch_per_epoch == 0:
                g_model.save(f'generator_{(i+1)/batch_per_epoch:03d}.h5')


if __name__ == '__main__':
    main_gan_test()
