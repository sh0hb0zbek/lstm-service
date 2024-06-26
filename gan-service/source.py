import numpy
from numpy import zeros
from numpy import ones
from numpy import linspace
from numpy import asarray
from numpy import arccos
from numpy import clip
from numpy import dot
from numpy import sin
from numpy import vstack
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from numpy.random import randn
from numpy.random import randint
from numpy.linalg import norm
from keras import backend
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.constraints import Constraint
from skimage.transform import resize
from scipy.linalg import sqrtm
from layers import layer

def main_gan(args):
    """
    argv_discriminator = {
                         'model_type': 'Model',
                         'layers': [layers.layer(...),
                                    layers.layer(...),
                                    layers.layer(...),
                                    layers.layer(...),
                                    ...],
                         'optimizer': layers.optimzer(...),
                         'loss': '...',
                         'metrics': '...'
    }
    
    argv_generator = {
                     'model_type': 'Model',
                     'layers': [layers.layer(...),
                                layers.layer(...),
                                layers.layer(...),
                                layers.layer(...),
                                ...],
                     'optimizer': layers.optimzer(...),
                     'loss': '...',
                     'metrics': '...',
                     'do_compile': False  # --> if model is generator
    }
    
    argv_gan = {
                'optimizer': layers.optimizer(...),
                'loss': ...,
                'model_type': 'Sequential' # or 'Model'
    }
    
    argv_train = {
                  'train': train, # train function
                  'n_epochs': ...,
                  'n_batch': ...
    }
    
    dataset --> training dataset
    """

    argv_discriminator = args[0]
    argv_generator = args[1]
    argv_gan = args[2]
    dataset = args[4]
    argv_train = args[3]
    train = argv_train.pop('train')
    
    d_model = define_model(**argv_discriminator)
    g_model = define_model(**argv_generator)
    gan_model = define_gan(g_model, d_model, **argv_gan)
    train(g_model, d_model, gan_model, dataset, **argv_train)

# function for scaling images
def scale_images(images, scale=[-1, 1]):
    # convert from unit8 to float32
    converted_images = images.astype('float32')
    # scale
    min_value = converted_images.max()
    max_value = converted_images.min()
    average_value = (max_value - min_value) / (scale[1] - scale[0])
    converted_images = (images - min_value) / average_value + scale[0]
    return converted_images

# function for generating points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space and reshape into a batch of inputs for the network
    return randn(latent_dim * n_samples).reshape((n_samples, latent_dim))

# define the combined generator and discriminator model, for updating generator
def define_gan(g_model, d_model, optimizer, loss='binary_crossentropy', model_type='Sequential'):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    if model_type == 'Sequential':
        # connect them
        model = Sequential()
        # add generator
        model.add(g_model)
        # add discriminator
        model.add(d_model)
    elif model_type == 'Model':
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get image output from the generator model
        gen_output = g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gan_noise, gen_label], gan_output)
    # compile model
    model.compile(loss=loss, optimizer=optimizer)
    return model

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    x = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return x, y

# generate a batch of images, returns images and targets
def generate_fake_samples_2(g_model, dataset, path_shape):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

# select real samples from dataset
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    x = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return x, y

# select a batch of random samples, returns images and target
def generate_real_samples_2(dataset, n_samples, patch_shape):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (0)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y

# uniform interpolate between two points in latent space
def interpolate_points(p1, p2, intrpl, n_steps=10):
    # interpolate ratios betweem the points
    ratios = linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in rations:
        v = intrpl(ratio, p1, p2)
        vectors.append(v)
    return asarray(vectors)

# spherical linear interpolation (slerp)
def slerp(val, low, high):
    omega = arccos(clip(dot(low/norm(low), high/norm(high)), -1, 1))
    so = sin(omega)
    if so == 0:
        # L'Hopital's rule/LERP
        return (1.0-val)*low + val*high
    return sin((1.0-val)*omega) / so*low + sin(val*omega) / so * high

# uniform interpolation (uni_inter)
def uni_inter(val, low, high):
    return (1.0-val)*p1 + val*p2

# average list of latent space vectors
def average_points(points, ix):
    # convert to zero offset points
    zero_ix = [i-1 for i in ix]
    # retreive required points
    vectors = points[zero_ix]
    # average the vectors
    avg_vector = mean(vectors, axis=0)
    # combine original and avg vectors
    all_vectors = vstack((vectors, avg_vector))
    return all_vectors

# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1e-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx*(log(p_yx+eps)-log(p_y+eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbot interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

# assumes images have any shape and pixels in [0, 255]
def calculate_inception_score(images, n_split=10, eps=1e-16):
    # load inception v3 model
    model = InceptionV3()
    # enumerate splits of images/predictions
    scores = list()
    n_part = floor(images.shape[0]/n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i*n_part, (i+1)*n_part
        subset = images[ix_start:ix_end]
        # convert from unit8 to float32
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299, 299, 3))
        # pre-process images, scale [-1, 1]
        sunset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = expand_dim(p_yx.mean(axis=0), 0)
        # calculate KL divergence using log probabilities
        kl_d = p_yx*(log(p_yx+eps)-log(p_y+eps))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = mean(sum_kl_d)
        # undo the log
        is_score = exp(avg_kl_d)
        # store
        scores.append(is_score)
    # average across images
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

# calculate frechet inception distance (FID) using NumPy
def calculate_fid_w_numpy(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# calculate frechet inception distance (FID) using inception v3 model (keras)
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # use calculate_fid_w_numpy() to calculate FID
    return calculate_fid_w_numpy(act1, act2)

# clip model weights to a given hypercube (constraint)
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)

# define a model
def define_model(lyrs, model_type='Sequential', optimizer='rmsprop', loss=None, metrics=None, do_compile=True):
    if model_type == 'Sequential':
        model = Sequential()
        for lyr in lyrs:
            model.add(lyr)
    elif model_type == 'Model':
        if 'Concatenate' in lyrs:
            idx = lyrs.index('Concatenate')
            idx_end = lyrs.index('End')
            in_label = lyrs[0]
            li = in_label
            for i in range(1, idx):
                li = lyrs[i](li)
            in_image = lyrs[idx+1]
            lii = in_image
            for i in range(idx+2, idx_end):
                lii = lyrs[i](lii)
            merge = layer('Concatenate')([lii, li])
            fe = merge
            for i in range(idx_end+1, len(lyrs)):
                fe = fe(lyrs[i])
            inputs = [in_image, in_label]
            outputs = fe
        else:
            in_image = layers[0]
            li = in_image
            for i in range(1, len(layers)):
                li = layers[i](li)
            inputs = in_image
            outputs = li
        model = Model(inputs=inputs, outputs=outputs)
    if do_compile:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# train the generator and discriminator models
def train_01(g_model, d_model, gan_model, dataset, laten_dim,
             n_epochs=100, n_batch=128, do_print=True, save_model=True):
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(batch_per_epoch):
            # get randomly selected 'real' samples
            x_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(x_real, y_real)
            # generate 'fake' samples
            x_fake, y_fake = generate_fake_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(x_fake, y_fake)
            # prepare points in latent space as input for the generator
            x_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(x_gan, y_gan)
            if do_print:
                print(f'>{i+1:03d} {j+1:03d}/{n_batch}, d1={d_loss1:.3f}, d2{d_loss2:.3f}, g={g_loss:.3f}')
        # save the generator model
        if save_model:
            g_model.save(f'generator_{i+1:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train the generator and discriminator
def train_15(g_model, d_model, gan_model, dataset, latent_dim,
             n_epochs=20, n_batch=64, do_print=False, save_model=True):
    # calculate the number of batches per training epoch
    bath_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bath_per_epoch * n_epochs
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
                g_model.save(f'generator_{(i+1)//batch_per_epoch:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train the generator and critic
def train_16(g_model, c_model, gan_model, dataset, latent_dim,
             n_epochs=10, n_batch=64, n_critic=5, do_print=False, save_model=True):
    # calculate the number of batches per training epoch
    bath_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bath_per_epoch * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update critic model weights
            c_loss1 = c_model.train_on_batch(X_real, y_real)
            c1_tmp.append(c_loss1)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update critic model weights
            c_loss2 = c_model.train_on_batch(X_fake, y_fake)
            c2_tmp.append(c_loss2)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        X_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic✬s error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        g_hist.append(g_loss)
        if do_print:
            print(f'>{i+1:03d}, c1={c1_hist[-1]:.3f}, c2={c2_hist[-1]:.3f}, g={g_loss:.3f}')
        # save the generator model
        if save_model:
            if (i+1)%batch_per_epoch == 0:
                g_model.save(f'generator_{(i+1)//batch_per_epoch:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train the generator and discriminator
def train_17(g_model, d_model, gan_model, dataset, latent_dim,
             n_epochs=100, n_batch=128, do_print=False, save_model=True):
    bath_per_epoch = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bath_per_epoch):
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
            g_model.save(f'generator_{i+1:03d}.h5')
# --------------------------------------------------------------------------------------------------
# train the generator and discriminator
def train_18(g_model, d_model, gan_model, dataset, latent_dim, n_cat,
             n_epochs=100, n_batch=64, do_print=False, save_model=True):
    # calculate the number of batches per training epoch
    batch_per_epoch = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected ✬real✬ samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # update discriminator and q model weights
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        # generate ✬fake✬ examples
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_cat, half_batch)
        # update discriminator model weights
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # prepare points in latent space as input for the generator
        z_input, cat_codes = generate_latent_points(latent_dim, n_cat, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the g via the d and q error
        _,g_1,g_2 = gan_model.train_on_batch(z_input, [y_gan, cat_codes])
        if do_print:
            print(f'>{i+1:03d}, d[{d_loss1:.3f},{d_loss2:.3f}], g[{g_1:.3f}] q[{g_2:.3f}]')
        # save the generator model
        if save_model:
            if (i+1)%batch_per_epoch == 0:
                g_model.save(f'generator_{(i+1)/batch_per_epoch:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train the generator and discriminator
def train_19(g_model, d_model, gan_model, dataset, latent_dim,
             n_epochs=100, n_batch=64, do_print=False, save_model=True):
    # calculate the number of batches per training epoch
    batch_per_epoch = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # get randomly selected ✬real✬ samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        _,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
        # generate ✬fake✬ examples
        [X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        _,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
        # prepare points in latent space as input for the generator
        [z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator✬s error
        _,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
        if do_print:
            print(f'>{i+1:03d}, dr[{d_r1:.3f},{d_r2:.3f}], df[{d_f:.3f},{d_f2:.3f}], g[{g_1:.3f},{g_2:.3f}]')
        # save the generator model
        if save_model:
            if (i+1)%batch_per_epoch == 0:
                g_model.save(f'generator_{(i+1)/batch_per_epoch:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train the generator and discriminator
def train_20(g_model, d_model, c_model, gan_model, dataset, latent_dim,
          n_epochs=20, n_batch=100, do_print=False, save_model=True):
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset)
    print(X_sup.shape, y_sup.shape)
    # calculate the number of batches per training epoch
    batch_per_epoch = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print(f'n_epochs={n_epochs}, n_batch={n_batch}, 1/2={half_batch}, b/e={batch_per_epoch}, steps={n_steps}')
    # manually enumerate epochs
    for i in range(n_steps):
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        if do_print:
            print(f'>{i+1}, c[{c_loss:.3f},{c_acc*100:.0f}], d[{d_loss1:.3f},{d_loss2:.3f}], g[{g_loss:.3f}]')
        # save the generator model
        if save_model:
            if (i+1)%bath_per_epoch == 0:
                g_model.save(f'generator_{(i+1)/batch_per_epoch:03d}.h5')

# --------------------------------------------------------------------------------------------------
# train pix2pix models
def train_22(d_model, g_model, gan_model, dataset,
          n_epochs=100, n_batch=1, do_print=False, save_model=True):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bath_per_epoch = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bath_per_epovh * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        if do_print:
            print(f'>{i+1:03d}, d1[{d_loss1:.3f}] d2[{d_loss2:.3f}] g[{g_loss:.3f}]')
        # save the generator model
        if save_model:
            if (i+1)%bath_per_epoch:
                g_model.save(f'generator_{(i+1)/batch_per_epoch}.h5')

# --------------------------------------------------------------------------------------------------
# train cyclegan models
def train_26(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset,
            n_epochs=100, n_batch=1, do_print=False, save_model=True):
    # determine the output square shape of the discriminator
    n_patch = d_model_A.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch)
        X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB,
        X_realB, X_realA, X_realB])
        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
        if do_print:
            print(f'>{i+1}, dA[{dA_loss1:.3f},{dA_loss2:.3f}] dB[{dB_loss1:.3f},{dB_loss2:.3f}], g[{g_loss1:.3f},{g_loss2:.3f}]')
        # save the generator model
        if save_model:
            if (i+1)%bath_per_epoch:
                g_model_AtoB.save(f'generator_AtoB_{(i+1)/batch_per_epoch}.h5')
                g_model_BtoA.save(f'generator_AtoB_{(i+1)/batch_per_epoch}.h5')

# --------------------------------------------------------------------------------------------------

