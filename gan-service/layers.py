# define an optimizer for model compilation
def optimizer(name, **kwargs):
    from tensorflow.keras import optimizers
    if name == 'Adam':
        return optimizers.Adam(**kwargs)
    elif name == 'Adadelta':
        return optimizers.Adadelta(**kwargs)
    elif name == 'Adagrad':
        return optimizers.Adagrad(**kwargs)
    elif name == 'Adamax':
        return optimizers.Adamax(**kwargs)
    elif name == 'Ftrl':
        return optimizers.Ftrl(**kwargs)
    elif name == 'Nadam':
        return optimizers.Nadam(**kwargs)
    elif name == 'RMSprop':
        return optimizers.RMSprop(**kwargs)
    elif name == 'SGD':
        return optimizers.SGD(**kwargs)
    else:
        return None


# define layer weight regularizers
def regularizer(reg_name, **kwargs):
    from keras import regularizers
    if reg_name == 'l1':
        return regularizers.L1(**kwargs)
    elif reg_name == 'l2':
        return regularizers.L2(**kwargs)
    elif reg_name == 'l1l2':
        return regularizers.L1L2(**kwargs)
    else:
        return None


# define layer weight initializer
def initializer(init_name, **kwargs):
    from keras import initializers
    if init_name == 'RandomNormal':
        return initializers.RandomNormal(**kwargs)
    elif init_name == 'RandomUniform':
        return initializers.RandomUniform(**kwargs)
    elif init_name == 'TruncatedNormal':
        return initializers.TruncatedNormal(**kwargs)
    elif init_name == 'Zeros':
        return initializers.Zeros(**kwargs)
    elif init_name == 'Ones':
        return initializers.Ones(**kwargs)
    elif init_name == 'GlorotNormal':
        return initializers.GlorotNormal(**kwargs)
    elif init_name == 'GlorotUniform':
        return initializers.GlorotUniform(**kwargs)
    elif init_name == 'HeNormal':
        return initializers.HeNormal(**kwargs)
    elif init_name == 'HeUniform':
        return initializers.HeUniform(**kwargs)
    elif init_name == 'Identity':
        return initializers.Identity(**kwargs)
    elif init_name == 'Orthogonal':
        return initializers.Orthogonal(**kwargs)
    elif init_name == 'Constant':
        return initializers.Constant(**kwargs)
    elif init_name == 'VarianceScaling':
        return initializers.VarianceScaling(**kwargs)
    else:
        return None


# define layer weight constraints
def constraint(const_name, **kwargs):
    from keras import constraints
    if const_name == 'MaxNorm':
        return constraints.MaxNorm(**kwargs)
    elif const_name == 'MinMaxNorm':
        return constraints.MinMaxNorm(**kwargs)
    elif const_name == 'NonNeg':
        return constraints.NonNeg(**kwargs)
    elif const_name == 'UnitNorm':
        return constraints.UnitNorm(**kwargs)
    else:
        return None


# define keras layers
def layer(layer_name, **kwargs):
    from tensorflow.keras import layers
    # reccurrent layers (RNN)
    if layer_name == 'LSTM':
        return layers.LSTM(**kwargs)
    elif layer_name == 'GRU':
        return layers.GRU(**kwargs)
    elif layer_name == 'SimpleRNN':
        return layers.SimpleRNN(**kwargs)
    elif layer_name == 'TimeDistributed':
        return layers.TimeDistributed(**kwargs)
    elif layer_name == 'Bidirectional':
        return layers.Bidirectional(**kwargs)
    elif layer_name == 'ConvLSTM2D':
        return layers.ConvLSTM2D(**kwargs)
    elif layer_name == 'RNN':
        return layers.RNN(**kwargs)
    # convolution layers (CNN)
    elif layer_name == 'Conv1D':
        return layers.Conv1D(**kwargs)#filters=16, kernel_size=(1, ))
    elif layer_name == 'Conv2D':
        return layers.Conv2D(**kwargs)
    elif layer_name == 'Conv3D':
        return layers.Conv3D(**kwargs)
    elif layer_name == 'SeparableConv1D':
        return layers.SeparableConv1D(**kwargs)
    elif layer_name == 'SeparableConv2D':
        return layers.SeparableConv2D(**kwargs)
    elif layer_name == 'DepthwiseConv2D':
        return layers.DepthwiseConv2D(**kwargs)
    elif layer_name == 'Conv2DTranspose':
        return layers.Conv2DTranspose(**kwargs)
    elif layer_name == 'Conv3DTranspose':
        return layers.Conv3DTranspose(**kwargs)
    # activation layers
    elif layer_name == 'ReLU':
        return layers.ReLU(**kwargs)
    elif layer_name == 'Softmax':
        return layers.Softmax(**kwargs)
    elif layer_name == 'LeakyReLU':
        return layers.LeakyReLU(**kwargs)
    elif layer_name == 'PReLU':
        return layers.PReLU(**kwargs)
    elif layer_name == 'ELU':
        return layers.ELU(**kwargs)
    elif layer_name == 'ThresholdedReLU':
        return layers.ThresholdedReLU(**kwargs)
    # pooling layers
    elif layer_name == 'MaxPooling1D':
        return layers.MaxPooling1D(**kwargs)
    elif layer_name == 'MaxPooling2D':
        return layers.MaxPooling2D(**kwargs)
    elif layer_name == 'MaxPooling3D':
        return layers.MaxPooling3D(**kwargs)
    elif layer_name == 'AveragePooling1D':
        return layers.AveragePooling1D(**kwargs)
    elif layer_name == 'AveragePooling2D':
        return layers.AveragePooling2D(**kwargs)
    elif layer_name == 'AveragePooling3D':
        return layers.AveragePooling3D(**kwargs)
    elif layer_name == 'GlobalMaxPooling1D':
        return layers.GlobalMaxPooling1D(**kwargs)
    elif layer_name == 'GlobalMaxPooling2D':
        return layers.GlobalMaxPooling2D(**kwargs)
    elif layer_name == 'GlobalMaxPooling3D':
        return layers.GlobalMaxPooling3D(**kwargs)
    elif layer_name == 'GlobalAveragePooling1D':
        return layers.GlobalAveragePooling1D(**kwargs)
    elif layer_name == 'GlobalAveragePooling2D':
        return layers.GlobalAveragePooling2D(**kwargs)
    elif layer_name == 'GlobalAveragePooling3D':
        return layers.GlobalAveragePooling3D(**kwargs)
    # core layers
    elif layer_name == 'Input':
        from keras import Input
        return Input(**kwargs)
    elif layer_name == 'Dense':
        return layers.Dense(**kwargs)
    elif layer_name == 'Activation':
        return layers.Activation(**kwargs)
    elif layer_name == 'Embedding':
        return layers.Embedding(**kwargs)
    elif layer_name == 'Masking':
        return layers.Masking(**kwargs)
    elif layer_name == 'Lambda':
        return layers.Lambda(**kwargs)
    # normalization layers
    elif layer_name == 'BatchNormalization':
        return layers.BatchNormalization(**kwargs)
    elif layer_name == 'LayerNormalization':
        return layers.LayerNormalization(**kwargs)
    # regularization layers
    elif layer_name == 'Dropout':
        return layers.Dropout(**kwargs)
    elif layer_name == 'SpatialDropout1D':
        return layers.SpatialDropout1D(**kwargs)
    elif layer_name == 'SpatialDropout2D':
        return layers.SpatialDropout2D(**kwargs)
    elif layer_name == 'SpatialDropout3D':
        return layers.SpatialDropout3D(**kwargs)
    elif layer_name == 'GaussianDropout':
        return layers.GaussianDropout(**kwargs)
    elif layer_name == 'GaussianNoise':
        return layers.GaussianNoise(**kwargs)
    elif layer_name == 'ActivityRegularization':
        return layers.ActivityRegularization(**kwargs)
    elif layer_name == 'AlphaDropout':
        return layers.AlphaDropout(**kwargs)
    # attention layers
    elif layer_name == 'MultiHeadAttention':
        return layers.MultiHeadAttention(**kwargs)
    elif layer_name == 'Attention':
        return layers.Attention(**kwargs)
    elif layer_name == 'AdditiveAttention':
        return layers.AdditiveAttention(**kwargs)
    # reshaping layers
    elif layer_name == 'Reshape':
        return layers.Reshape(**kwargs)
    elif layer_name == 'Flatten':
        return layers.Flatten(**kwargs)
    elif layer_name == 'RepeatVector':
        return layers.RepeatVector(**kwargs)
    elif layer_name == 'Permute':
        return layers.Permute(**kwargs)
    elif layer_name == 'Cropping1D':
        return layers.Cropping1D(**kwargs)
    elif layer_name == 'Cropping2D':
        return layers.Cropping2D(**kwargs)
    elif layer_name == 'Cropping3D':
        return layers.Cropping3D(**kwargs)
    elif layer_name == 'UpSampling1D':
        return layers.UpSampling1D(**kwargs)
    elif layer_name == 'UpSampling2D':
        return layers.UpSampling2D(**kwargs)
    elif layer_name == 'UpSampling3D':
        return layers.UpSampling3D(**kwargs)
    elif layer_name == 'ZeroPadding1D':
        return layers.ZeroPadding1D(**kwargs)
    elif layer_name == 'ZeroPadding2D':
        return layers.ZeroPadding2D(**kwargs)
    elif layer_name == 'ZeroPadding3D':
        return layers.ZeroPadding3D(**kwargs)
    # merging layers
    elif layer_name == 'Concatenate':
        return layers.Concatenate(**kwargs)
    elif layer_name == 'Average':
        return layers.Average(**kwargs)
    elif layer_name == 'Maximum':
        return layers.Maximum(**kwargs)
    elif layer_name == 'Minimum':
        return layers.Minimum(**kwargs)
    elif layer_name == 'Add':
        return layers.Add(**kwargs)
    elif layer_name == 'Subtract':
        return layers.Subtract(**kwargs)
    elif layer_name == 'Multiply':
        return layers.Multiply(**kwargs)
    elif layer_name == 'Dot':
        return layers.Dot(**kwargs)
    # locally connecteg layers
    elif layer_name == 'LocallyConnected1D':
        return layers.LocallyConnected1D(**kwargs)
    elif layer_name == 'LocallyConnected2D':
        return layers.LocallyConnected2D(**kwargs)
    else:
        return None
