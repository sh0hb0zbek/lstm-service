import tensorflow as tf
from matplotlib import pyplot
import numpy as np
import sys
import h5py
from tqdm import tqdm
from IPython import display as ipythondisplay


class TrainingDatasetLoader(object):
    def __init__(self, data_path):
        # print(f'Opening {data_path}')
        self.cache = h5py.File(data_path, 'r')
        # print('Loading data into memory ...')
        sys.stdout.flush()
        self.images = self.cache['images'][:]
        self.labels = self.cache['labels'][:].astype(np.float32)
        self.image_dims = self.images.shape
        n_train_samples = self.image_dims[0]
        
        self.train_inds = np.random.permutation(np.arange(n_train_samples))
        
        self.pos_train_inds = self.train_inds[self.labels[self.train_inds, 0] == 1.0]
        self.neg_train_inds = self.train_inds[self.labels[self.train_inds, 0] != 1.0]
    
    def get_train_size(self):
        return self.train_inds.shape[0]
    
    def get_train_steps_per_epoch(self, batch_size, factor=10):
        return self.get_train_size()//factor//batch_size
    
    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None,
                  return_inds=False):
        if only_faces:
            selected_inds = np.random_choice(
                self.pos_train_inds, size=n, replace=False, p=p_pos)
        else:
            selected_pos_inds = np.random.choice(
                self.pos_train_inds, size=n//2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(
                self.neg_train_inds, size=n//2, replace=False, p=p_neg)
            selected_inds = np.concatenate((selected_pos_inds, selected_neg_inds))
        sorted_inds = np.sort(selected_inds)
        train_img = (self.images[sorted_inds,:,:,::-1]/255.).astype(np.float32)
        train_label = self.labels[sorted_inds, ...]
        return (train_img, train_label, sorted_inds) if return_inds \
    else (train_img, train_label)
    
    def get_n_most_prob_faces(self, prob, n):
        idx = np.argsort(prob)[::-1]
        most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
        return (self.images[most_prob_inds,...]/255.).astype(np.float32)
    
    def get_all_train_faces(self):
        return self.images[self.pos_train_inds]


class LossHistory:
    def __init__(self, smoothing_factor=0.0):
        self.alpha = smoothing_factor
        self.loss = list()
    def append(self, value):
        self.loss.append(self.alpha*self.loss[-1] + (1-self.alpha)*value \
                        if len(self.loss)>0 else value)
    def get(self):
        return self.loss


class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale

        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            pyplot.cla()

            if self.scale is None:
                pyplot.plot(data)
            elif self.scale == 'semilogx':
                pyplot.semilogx(data)
            elif self.scale == 'semilogy':
                pyplot.semilogy(data)
            elif self.scale == 'loglog':
                pyplot.loglog(data)
            else:
                raise ValueError("unrecognized parameter scale {}".format(self.scale))

            pyplot.xlabel(self.xlabel); pyplot.ylabel(self.ylabel)
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(pyplot.gcf())

            self.tic = time.time()


def plot_sample(x, y, vae):
    pyplot.figure(figsize=(2,1))
    pyplot.subplot(1, 2, 1)

    idx = np.where(y==1)[0][0]
    pyplot.imshow(x[idx])
    pyplot.grid(False)

    pyplot.subplot(1, 2, 2)
    _, _, _, recon = vae(x)
    recon = np.clip(recon, 0, 1)
    pyplot.imshow(recon[idx])
    pyplot.grid(False)

    pyplot.show()


def define_classifier(n_outputs=1, n_filters=8, kernel_size=3, strides=2, padding='same',
                      activation='relu', n_conv_layers=4, dense_units=[512]):
    model = tf.keras.Sequential()
    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv2D(filters=pow(2,i)*n_filters, kernel_size=kernel_size,
                                         strides=strides, padding=padding, activation=activation))
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    for units in dense_units:
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
    model.add(tf.keras.layers.Dense(units=n_outputs, activation='softmax'))
    return model


def define_decoder_network(n_filters=8, kernel_size=3, strides=2, padding='same',
                                activation='relu', n_conv_layers=3):
    
    decoder = tf.keras.Sequential()
    decoder.add(tf.keras.layers.Dense(units=4*4*(pow(2,n_conv_layers)*n_filters), activation='relu'))
    decoder.add(tf.keras.layers.Reshape(target_shape=(4, 4, pow(2,n_conv_layers)*n_filters)))
    for i in range(n_conv_layers-1, -1, -1):
        decoder.add(tf.keras.layers.Conv2DTranspose(filters=pow(2, i)*n_filters, kernel_size=kernel_size,
                                                    strides=strides, padding=padding, activation=activation))
    decoder.add(tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=kernel_size, strides=strides,
                                                padding=padding, activation=activation))
    return decoder


### Defining the VAE loss function ###
def vae_loss_function(x, x_recon, mu, logsigma, kl_weights=5e-4):
    ''' 
    Function to calculate VAE loss given:
          an input x, 
          reconstructed output x_recon, 
          encoded means mu, 
          encoded log of standard deviation logsigma, 
          weight parameter for the latent loss kl_weight
    '''
    latent_loss = 0.5 / tf.reduce_sum(tf.exp(logsigma)+tf.square(mu)-1.0-logsigma)
    reconstruction_loss = tf.reduce_mean(tf.abs(x-x_recon), axis=(1,2,3))
    vae_loss = kl_weights*latent_loss+reconstruction_loss
    return vae_loss


### VAE Reparameterization ###
def sampling(z_mean, z_logsigma):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            z_mean, z_logsigma (tensor): mean and log of standard deviation of latent distribution (Q(z|X))
        # Returns
            z (tensor): sampled latent vector
    """
    batch, latent_dim = z_mean.shape
    epsilon = tf.random.normal(shape=(batch, latent_dim))
    z = z_mean + tf.math.exp(0.5 * z_logsigma) * epsilon
    return z


### Loss function for DB-VAE ###
def debiasing_loss_function(x, x_pred, y, y_logit, mu, logsigma):
    """
    Loss function for DB-VAE.
        # Arguments
            x: true input x
            x_pred: reconstructed x
            y: true label (face or not face)
            y_logit: predicted labels
            mu: mean of latent distribution (Q(z|X))
            logsigma: log of standard deviation of latent distribution (Q(z|X))
        # Returns
            total_loss: DB-VAE total loss
            classification_loss = DB-VAE classification loss
    """
    vae_loss = vae_loss_function(x, x_pred, mu, logsigma)
    classification_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_logit)
    face_indicator = tf.cast(tf.equal(y, 1), tf.float32)
    total_loss = tf.reduce_mean(classification_loss + face_indicator * vae_loss)
    return total_loss, classification_loss


### defining and creating the DB-VAE ###
class DB_VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, latent_dim):
        super(DB_VAE, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = encoder
        self.decoder = decoder
    
    # function to feed images into encoder, encode the latent space, and output
    # classification probability
    def encode(self, x):
        # encoder output
        encoder_output = self.encoder(x)
        
        # classification prediction
        y_logit = tf.expand_dims(encoder_output[:,0], -1)
        # latent variable distribution parameters
        z_mean = encoder_output[:, 1:self.latent_dim+1]
        z_logsigma = encoder_output[:, self.latent_dim+1:]
        
        return y_logit, z_mean, z_logsigma
    
    # VAE reparameterization: given a mean and logsigma, sample latent variables
    def reparameterize(self, z_mean, z_logsigma):
        z = sampling(z_mean, z_logsigma)
        return z
    
    # decode the latent space and outpuy reconstruction
    def decode(self, z):
        recunstruction = self.decoder(z)
        return recunstruction
    
    # the call function will be used to pass inputs x through the core VAE
    def call(self, x):
        # encode input to a prediction and latent space
        y_logit, z_mean, z_logsigma = self.encode(x)
        
        # reparameterization
        z = self.reparameterize(z_mean, z_logsigma)
        
        # reconstruction
        recon = self.decode(z)
        
        return y_logit, z_mean, z_logsigma, recon
    
    # predict face or not face logit for given input x
    def predict(self, x):
        y_logit, z_mean, z_logsigma = self.encode(x)
        return y_logit


# function to return the means for an input image batch
def get_latent_mu(images, dbvae, batch_size=1024):
    N = images.shape[0]
    mu = np.zeros((N, dbvae.latent_dim))
    for start_ind in range(0, N, batch_size):
        end_ind = min(start_ind+batch_size, N+1)
        batch = (images[start_ind:end_ind]).astype(np.float32)/255.
        _, batch_mu, _ = dbvae.encode(batch)
        mu[start_ind:end_ind] = batch_mu
    return mu


### Resampling algorithm for DB-VAE ###
def get_training_sample_probabilities(images, dbvae, bins=10, smoothing_fac=1e-3):
    '''
    Function that recomputes the sampling probabilities for images within a batch
    based on how they distribute across the training data
    '''
    # print('Recomputing the sampling probabilities')
    
    # run the input batch and get the latent variable means
    mu = get_latent_mu(images, dbvae)
    
    # sampling probabilities for the images
    training_sample_p = np.zeros(mu.shape[0])
    
    # consider the distribution for each latent variable
    for i in range(dbvae.latent_dim):
        latent_distribution = mu[:,i]
        
        # generate a histogram of the latent distibution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)
        
        # find which latent bin every data smaple falls
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')
        
        # call the digitize funciton to find which bins in the latent distribution
        # every data sample falls in to
        bin_idx = np.digitize(latent_distribution, bin_edges)
        
        # smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density /= np.sum(hist_smoothed_density)
        
        # invert the density function
        p = 1.0/(hist_smoothed_density[bin_idx-1])
        
        # normalizer all probabilities
        p /= np.sum(p)
        
        # update sampling probabilities by considering whether the newly computed
        # p is greater than the existinf sampling probabilities
        training_sample_p = np.maximum(p, training_sample_p)
    
    # final normalization
    training_sample_p /= np.sum(training_sample_p)
    return training_sample_p


@tf.function
def train_step(x, y, dbvae_model, optimizer):
    with tf.GradientTape() as tape:
        # feed input x into dbvae. Note that this is using the DB_VAE call function
        y_logit, z_mean, z_logsigma, x_recon = dbvae_model(x)
        
        # call the DB_VAE loss function to compute the loss
        loss, class_loss = debiasing_loss_function(x, x_recon, y, y_logit, z_mean, z_logsigma)
    # use the `GradientTaoe.gradient` method to compute the gradients
    grads = tape.gradient(loss, dbvae_model.trainable_variables)
    
    # apply gradients to variables
    optimizer.apply_gradients(zip(grads, dbvae_model.trainable_variables))
    return loss


def dbvae_train(dbvae_model, optimizer, train_dataset, dataset_loader, epochs=10, batch_size=32,
                learning_rate=1e-4, latent_dim=100, do_display=False):
    for i in range(epochs):
        if do_display:
            ipythondisplay.clear_output(wait=True)
            print(f'Starting epoch {i+1:2d}/{epochs}')
        # recompute data sampling probabilities
        p_faces = get_training_sample_probabilities(train_dataset, dbvae_model)
        
        # get a batch of training data and compute the training step
        loop = range(dataset_loader.get_train_size()//batch_size)
        if do_display: loop = tqdm(loop)
        for j in loop:
            # load a batch of data
            (x, y) = dataset_loader.get_batch(batch_size, p_pos=p_faces)
            # loss optimization
            loss = train_step(x, y, dbvae_model, optimizer)
            
            if j%500==0  and do_display:
                plot_sample(x, y, dbvae_model)
