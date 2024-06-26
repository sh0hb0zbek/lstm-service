from source import *
import tensorflow as tf

def test():
    # load training dataset
    path_to_training_data = tf.keras.utils.get_file('train_face.py', 'https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1')
    loader = TrainingDatasetLoader(path_to_training_data)
    # get training faces from data loader
    all_faces = loader.get_all_train_faces()
    
    # hyperparameters
    batch_size    = 64
    learning_rate = 1e-3
    latent_dim    = 100
    epochs        = 6
    encoder_dims  = 2*latent_dim+1
    
    # intantiate a new DB-VAE model and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # to use all available GPUs for training the model
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        encoder   = define_classifier(encoder_dims, n_filters=16, kernel_size=5)
        decoder   = define_decoder_network(n_filters=16)
        dbvae     = DB_VAE(encoder, decoder, latent_dim)

    # train the model
    dbvae_train(
        dbvae_model=dbvae,
        optimizer=optimizer,
        train_dataset=all_faces,
        dataset_loader=loader,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        latent_dim=latent_dim,
        do_display=False)


if __name__ == '__main__':
    test()
