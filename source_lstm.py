def normalizer(scl, values, inverse=False):
    if inverse:
        return scl.inverse_transform(values)
    return scl.transform(values)


def scaler(series, scaler_type='MinMaxScaler', feature_range=(0, 1)):
    if scaler_type in ['MMS', 'MinMaxScaler', 'MMScaler']:
        from sklearn.preprocessing import MinMaxScaler
        scl = MinMaxScaler(feature_range=feature_range).fit(series)
    elif scaler_type in ['SS', 'StandardScaler', 'SScaler']:
        from sklearn.preprocessing import StandardScaler
        scl = StandardScaler().fit(series)
    return scl


def encode(values, method='OneHotEncoder', sparse=False, categories='auto'):
    if method in ['LE', 'LabelEncoder', 'LEncoder', 'label_encoder', 'integer']:
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(values)
    elif method in ['OHE', 'OneHotEncoder', 'OHEncoder', 'one_hot_encoder', 'binary']:
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=sparse, categories=categories)
        encoded = encode(values=values, method='integer')
        encoded = encoded.reshape(len(encoded), 1)
        encoded = encoder.fit_transform(encoded)
    return encoded


def padding(sequence, method='pre', truncation=False, maxlen=2):
    from keras.preprocessing.sequence import pad_sequences
    if method == 'pre':
        if truncation:
            return pad_sequences(sequence, maxlen=maxlen)
        return pad_sequences(sequence)
    elif method == 'post':
        if truncation:
            return pad_sequences(sequence, maxlen=maxlen, truncating='post')
        return pad_sequences(sequence, padding='post')


def generate_dataset(n_samples, n_features, generate_type=None, centers=20, cluster_std=2,
                     noise=0.1, random_state=1):
    if generate_type is None:
        from random import randint
        return [randint(0, n_features - 1) for _ in range(n_samples)]
    elif generate_type == 'blobs':
        from sklearn.datasets import make_blobs
        x, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,
                          cluster_std=cluster_std, random_state=random_state)
    elif generate_type == 'regression':
        from sklearn.datasets import make_regression
        x, y = make_regression(n_samples=n_samples, n_features=n_features,
                               noise=noise, random_state=random_state)
    elif generate_type == 'circles':
        from sklearn.datasets import make_circles
        x, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    elif generate_type == 'moons':
        from sklearn.datasets import make_moons
        x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return x, y


def one_hot_encode(sequence, n_features):
    from numpy import array
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_features)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


def one_hot_decode(encoded_seq):
    from numpy import argmax
    return [argmax(vector) for vector in encoded_seq]


def layer(argument, base_layer=None):
    """
  LSTM
      - argument[1] - units /int/
      - argument[2] - return_sequences /boolean/
      - argument[3] - input_shape /tuple/
      - argument[4] - kernel_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[5] - bias_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[6] - recurrent_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[7] - kernel_constraint /list/ - [0]-constraint_type, [1]-value
      - argument[8] - bias_constraint /list/ - [0]-constraint_type, [1]-value
      - argument[9] - recurrent_constraint /float/
      e.g. ['LSTM', 15, True, (0, 1), ['l1', 0.01], None, None, 3.0, None, None]
  Dense
      - argument[1] - units /int/
      - argument[2] - input_dim /int/
      - argument[3] - activation /str/
      - argument[4] - kernel_initializer /str/
      - argument[5] - kernel_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[6] - bias_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[7] - kernel_constraint /list/ - [0]-constraint_type, [1]-value
      - argument[8] - bias_constraint /list/ - [0]-constraint_type, [1]-value
      e.g. ['Dense', 15, None, 'softmax', 'glorot_uniform', ['l1', 0.01], ['l1', 0.1], 3.0, 4.0]
  RepeatVector
      - argument[1] - n /int/
      e.g. ['RepeatVector', 3]
  TimeDistributed-Conv2D
      - argument[1] - layer /int/
      - argument[2] - kernel_size /tuple/
      - argument[3] - activation /str/
      - argument[4] - input_shape /tuple/
      - argument[5] - kernel_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[6] - bias_regularizer /list/ - [0]-regularizer_type, [1]-value
      - argument[7] - kernel_constraint /list/ - [0]-constraint_type, [1]-value
      - argument[8] - bias_constraint /list/ - [0]-constraint_type, [1]-value
      e.g. ['TimeDistributed-Conv2D', 75, (10, 1), 'relu', (3, 10), None, None, 3.0, 4.0]
  TimeDistributed-MaxPooling2D
      - argument[1] - pool_size /tuple/
      e.g. ['TimeDistributed-MaxPooling2D', (2, 2)]
  TimeDistributed-Flatten
      e.g. ['TimeDistributed-Flatten']
  Bidirectional-LSTM
      - argument[1] - units /int/
      - argument[2] - return_sequences /boolean/
      - argument[3] - input_shape /tuple/
      e.g. ['Bidirectional-LSTM', 15, True, (0, 1)]
  Activation
      - argument[1] - activation /str/
      e.g. ['Activation', 'relu']
  BatchNormalization
      e.g. ['BatchNormalization']
  Dropout
      - argument[1] - rate /float/ [0.0, 1.0]
      e.g. ['Dropout', 0.5]
  GaussianNoise
      - argument[1] - stddev /int or float/
      - argument[2] - seed /int/
      - argument[3] - input_shape /tuple/
      e.g. ['GaussianNoise', 0.1, (0, 1)]
  """
    if argument[0] == 'LSTM':
        from keras.layers import LSTM
        kernel_regularizer = argument[4]
        bias_regularizer = argument[5]
        recurrent_regularizer = argument[6]
        kernel_constraint = argument[7]
        bias_constraint = argument[8]
        recurrent_constraint = argument[9]
        if argument[4] is not None:
            kernel_regularizer = regularizer(argument[4][0], argument[4][1])
        if argument[5] is not None:
            bias_regularizer = regularizer(argument[5][0], argument[5][1])
        if argument[6] is not None:
            recurrent_regularizer = regularizer(argument[6][0], argument[6][1])
        if argument[7] is not None:
            kernel_constraint = constraint(argument[7][0], argument[7][1])
        if argument[8] is not None:
            bias_constraint = constraint(argument[8][0], argument[8][1])
        if argument[9] is not None:
            recurrent_constraint = constraint(argument[9][0], argument[9][1])
        if argument[3] is None:
            lyr = LSTM(units=argument[1], return_sequences=argument[2], kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer, recurrent_regularizer=recurrent_regularizer,
                       kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                       recurrent_constraint=recurrent_constraint)
        else:
            lyr = LSTM(units=argument[1], return_sequences=argument[2], kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer, recurrent_regularizer=recurrent_regularizer,
                       kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                       recurrent_constraint=recurrent_constraint, input_shape=argument[3])
    elif argument[0] == 'Dense':
        from keras.layers import Dense
        kernel_initializer = argument[4]
        kernel_regularizer = argument[5]
        bias_regularizer = argument[6]
        kernel_constraint = argument[7]
        bias_constraint = argument[8]
        if argument[4] is None:
            kernel_initializer = 'glorot_uniform'
        if argument[5] is not None:
            kernel_regularizer = regularizer(argument[5][0], argument[5][1])
        if argument[6] is not None:
            bias_regularizer = regularizer(argument[6][0], argument[6][1])
        if argument[7] is not None:
            kernel_constraint = constraint(argument[7][0], argument[7][1])
        if argument[8] is not None:
            bias_constraint = constraint(argument[8][0], argument[8][1])
        lyr = Dense(units=argument[1], input_dim=argument[2],
                    activation=argument[3], kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    elif argument[0] == 'RepeatVector':
        from keras.layers import RepeatVector
        lyr = RepeatVector(n=argument[1])
    elif argument[0] == 'Conv1D':
        from keras.layers import Conv1D
        lyr = Conv1D(argument[1], padding=argument[2], kernel_initializer=argument[3])
    elif argument[0] == 'Conv2D':
        from keras.layers import Conv2D
        kernel_regularizer = argument[5]
        bias_regularizer = argument[6]
        kernel_constraint = argument[7]
        bias_constraint = argument[8]
        if argument[5] is not None:
            kernel_regularizer = regularizer(argument[5][0], argument[5][1])
        if argument[6] is not None:
            bias_regularizer = regularizer(argument[6][0], argument[6][1])
        if argument[7] is not None:
            kernel_constraint = constraint(argument[7][0], argument[7][1])
        if argument[8] is not None:
            bias_constraint = constraint(argument[8][0], argument[8][1])
        lyr = Conv2D(filters=argument[1], kernel_size=argument[2],
                     activation=argument[3], input_shape=argument[4],
                     kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                     kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)
    elif argument[0] == 'MaxPooling2D':
        from keras.layers import MaxPooling2D
        lyr = MaxPooling2D(pool_size=argument[1], strides=argument[2], padding=argument[3])
    elif argument[0] == 'Flatten':
        from keras.layers import Flatten
        lyr = Flatten()
    elif 'TimeDistributed-' in argument[0]:
        argument[0] = argument[0][16:]
        from keras.layers import TimeDsitributed
        lyr = TimeDsitributed(layer(argument))
    elif 'Bidirectional-' in argument[0]:
        argument[0] = argument[0][14:]
        from keras.layers import Bidirectional
        lyr = Bidirectional(layer(argument))
    elif argument[0] == 'Activation':
        from keras.layers import Activation
        lyr = Activation(argument[1])
    elif argument[0] == 'BatchNormalization':
        from keras.layers import BatchNormalization
        lyr = BatchNormalization()
    elif argument[0] == 'Dropout':
        from keras.layers import Dropout
        lyr = Dropout(rate=argument[1])
    elif argument[0] == 'GaussianNoise':
        from keras.layers import GaussianNoise
        if argument[3] is not None:
            lyr = GaussianNoise(stddev=argument[1], seed=argument[2])
        else:
            lyr = GaussianNoise(stddev=argument[1], seed=argument[2], input_shape=argument[3])
    elif argument[0] == 'GlobalAveragePooling1D':
        from keras.layers import GlobalAveragePooling1D
        lyr = GlobalAveragePooling1D()
    elif argument[0] == 'Input':
        from keras.layers import Input
        lyr = Input(shape=(argument[1].shape[0], argument[1].shape[1]))
    elif argument[0] == 'Reshape':
        from keras.layers import Reshape
        lyr = Reshape(argument[1])
    elif argument[0] == 'DepthwiseConv2D':
        from keras.layers import DepthwiseConv2D
        lyr = DepthwiseConv2D(kernel_size=argument[1], depth_multiplier=argument[2],
                              data_format=argument[3], padding=argument[4])
    elif argument[0] == 'Permute':
        from keras.layers import Permute
        lyr = Permute(dims=argument[1])
    if base_layer is None:
        return lyr
    else:
        return lyr(base_layer)


def layer_dict(argument, base_layer=None):
    if argument['layer'] == 'LSTM':
        from keras.layers import LSTM
        argv = {'units': None, 'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'use_bias': True,
                'unroll': False,
                'kernel_initializer': 'glorot_uniform', 'recurrent_initializer': 'orthogonal', 'go_backwards': False,
                'bias_initializer': 'zeros', 'unit_forget_bias': True, 'kernel_regularizer': None, 'input_shape': None,
                'recurrent_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None,
                'time_major': False,
                'kernel_constraint': None, 'recurrent_constraint': None, 'bias_constraint': None,
                'return_sequences': False,
                'recurrent_dropout': 0.0, 'dropout': 0.0, 'return_state': False, 'stateful': False}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        if argv['kernel_regularizer'] is not None:
            argv['kernel_regularizer'] = regularizer(argv['kernel_regularizer'][0], argv['kernel_regularizer'][1])
        if argv['bias_regularizer'] is not None:
            argv['bias_regularizer'] = regularizer(argv['bias_regularizer'][0], argv['bias_regularizer'][1])
        if argv['recurrent_regularizer'] is not None:
            argv['recurrent_regularizer'] = regularizer(argv['recurrent_regularizer'][0],
                                                        argv['recurrent_regularizer'][1])
        if argv['activity_regularizer'] is not None:
            argv['activity_regularizer'] = regularizer(argv['activity_regularizer'][0], argv['activity_regularizer'][1])
        if argv['kernel_constraint'] is not None:
            argv['kernel_constraint'] = constraint(argv['kernel_constraint'][0], argv['kernel_constraint'][1])
        if argv['bias_constraint'] is not None:
            argv['bias_constraint'] = constraint(argv['bias_constraint'][0], argv['bias_constraint'][1])
        if argv['recurrent_constraint'] is not None:
            argv['recurrent_constraint'] = constraint(argv['recurrent_constraint'][0], argv['recurrent_constraint'][1])
        if argv['input_shape'] is None:
            lyr = LSTM(units=argv['units'], activation=argv['activation'], use_bias=argv['use_bias'],
                       unroll=argv['unroll'],
                       recurrent_activation=argv['recurrent_activation'],
                       recurrent_initializer=argv['recurrent_initializer'],
                       time_major=argv['time_major'], dropout=argv['dropout'],
                       kernel_regularizer=argv['kernel_regularizer'],
                       unit_forget_bias=argv['unit_forget_bias'], recurrent_regularizer=argv['recurrent_regularizer'],
                       stateful=argv['stateful'], return_state=argv['return_state'], go_backwards=argv['go_backwards'],
                       activity_regularizer=argv['activity_regularizer'], kernel_initializer=argv['kernel_initializer'],
                       recurrent_constraint=argv['recurrent_constraint'], bias_constraint=argv['bias_constraint'],
                       bias_initializer=argv['bias_initializer'], recurrent_dropout=argv['recurrent_dropout'],
                       return_sequences=argv['return_sequences'], kernel_constraint=argv['kernel_constraint'],
                       bias_regularizer=argv['bias_regularizer'])
        else:
            lyr = LSTM(units=argv['units'], activation=argv['activation'], use_bias=argv['use_bias'],
                       unroll=argv['unroll'],
                       recurrent_activation=argv['recurrent_activation'],
                       recurrent_initializer=argv['recurrent_initializer'],
                       time_major=argv['time_major'], dropout=argv['dropout'],
                       kernel_regularizer=argv['kernel_regularizer'],
                       unit_forget_bias=argv['unit_forget_bias'], recurrent_regularizer=argv['recurrent_regularizer'],
                       stateful=argv['stateful'], return_state=argv['return_state'], go_backwards=argv['go_backwards'],
                       activity_regularizer=argv['activity_regularizer'], kernel_initializer=argv['kernel_initializer'],
                       recurrent_constraint=argv['recurrent_constraint'], bias_constraint=argv['bias_constraint'],
                       bias_initializer=argv['bias_initializer'], recurrent_dropout=argv['recurrent_dropout'],
                       return_sequences=argv['return_sequences'], kernel_constraint=argv['kernel_constraint'],
                       bias_regularizer=argv['bias_regularizer'], input_shape=argv['input_shape'])
    elif argument['layer'] == 'Dense':
        from keras.layers import Dense
        argv = {'units': None, 'activation': None, 'use_bias': True, 'kernel_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
                'bias_constraint': None,
                'activity_regularizer': None, 'kernel_constraint': None, 'input_dim': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        if argv['kernel_regularizer'] is not None:
            argv['kernel_regularizer'] = regularizer(argv['kernel_regularizer'][0], argv['kernel_regularizer'][1])
        if argv['bias_regularizer'] is not None:
            argv['bias_regularizer'] = regularizer(argv['bias_regularizer'][0], argv['bias_regularizer'][1])
        if argv['kernel_constraint'] is not None:
            argv['kernel_constraint'] = constraint(argv['kernel_constraint'][0], argv['kernel_constraint'][1])
        if argv['bias_constraint'] is not None:
            argv['bias_constraint'] = constraint(argv['bias_constraint'][0], argv['bias_constraint'][1])
        if argv['input_dim'] is None:
            lyr = Dense(units=argv['units'], activation=argv['activation'],
                        activity_regularizer=argv['activity_regularizer'],
                        kernel_initializer=argv['kernel_initializer'], bias_initializer=argv['bias_initializer'],
                        kernel_regularizer=argv['kernel_regularizer'], bias_regularizer=argv['bias_regularizer'],
                        kernel_constraint=argv['kernel_constraint'], bias_constraint=argv['bias_constraint'],
                        use_bias=argv['use_bias'])
        else:
            lyr = Dense(units=argv['units'], activation=argv['activation'],
                        activity_regularizer=argv['activity_regularizer'],
                        kernel_initializer=argv['kernel_initializer'], bias_initializer=argv['bias_initializer'],
                        kernel_regularizer=argv['kernel_regularizer'], bias_regularizer=argv['bias_regularizer'],
                        kernel_constraint=argv['kernel_constraint'], bias_constraint=argv['bias_constraint'],
                        use_bias=argv['use_bias'], input_dim=argv['input_dim'])
    elif argument['layer'] == 'RepeatVector':
        from keras.layers import RepeatVector
        repeat_vector_argv = {'n': None}
        for key in argument.keys():
            if key in repeat_vector_argv.keys():
                repeat_vector_argv[key] = argument[key]
        lyr = RepeatVector(n=repeat_vector_argv['n'])
    elif argument['layer'] == 'Conv1D':
        from keras.layers import Conv1D
        argv = {'filters': None, 'kernel_size': None, 'strides': 1, 'padding': 'valid', 'dilation_rate': 1,
                'data_format': 'channels_last', 'groups': 1, 'activation': None, 'use_bias': True,
                'bias_constraint': None,
                'kernel_initializer': 'glorot_uniform', 'bias_initializer': 'zeros', 'kernel_regularizer': None,
                'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        if argv['kernel_regularizer'] is not None:
            argv['kernel_regularizer'] = regularizer(argv['kernel_regularizer'][0], argv['kernel_regularizer'][1])
        if argv['bias_regularizer'] is not None:
            argv['bias_regularizer'] = regularizer(argv['bias_regularizer'][0], argv['bias_regularizer'][1])
        if argv['activity_regularizer'] is not None:
            argv['activity_regularizer'] = regularizer(argv['activity_regularizer'][0], argv['activity_regularizer'][1])
        if argv['kernel_constraint'] is not None:
            argv['kernel_constraint'] = constraint(argv['kernel_constraint'][0], argv['kernel_constraint'][1])
        if argv['bias_constraint'] is not None:
            argv['bias_constraint'] = constraint(argv['bias_constraint'][0], argv['bias_constraint'][1])
        lyr = Conv1D(filters=argv['filters'], kernel_size=argv['kernel_size'], data_format=argv['data_format'],
                     dilation_rate=argv['dilation_rate'], activation=argv['activation'], use_bias=argv['use_bias'],
                     kernel_initializer=argv['kernel_initializer'], padding=argv['padding'], groups=argv['groups'],
                     bias_initializer=argv['bias_initializer'], bias_regularizer=argv['bias_regularizer'],
                     kernel_regularizer=argv['kernel_regularizer'], strides=argv['strides'],
                     kernel_constraint=argv['kernel_constraint'], bias_constraint=argv['bias_constraint'],
                     activity_regularizer=argv['activity_regularizer'])
    elif argument['layer'] == 'Conv2D':
        from keras.layers import Conv2D
        argv = {'filters': None, 'kernel_size': None, 'strides': (1, 1), 'padding': 'valid',
                'bias_initializer': 'zeros',
                'data_format': None, 'dilation_rate': (1, 1), 'groups': 1, 'activation': None, 'use_bias': True,
                'kernel_initializer': 'glorot_uniform', 'kernel_regularizer': None, 'bias_regularizer': None,
                'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'input_shape': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        if argv['kernel_regularizer'] is not None:
            argv['kernel_regularizer'] = regularizer(argv['kernel_regularizer'][0], argv['kernel_regularizer'][1])
        if argv['bias_regularizer'] is not None:
            argv['bias_regularizer'] = regularizer(argv['bias_regularizer'][0], argv['bias_regularizer'][1])
        if argv['activity_regularizer'] is not None:
            argv['activity_regularizer'] = regularizer(argv['activity_regularizer'][0], argv['activity_regularizer'][1])
        if argv['kernel_constraint'] is not None:
            argv['kernel_constraint'] = constraint(argv['kernel_constraint'][0], argv['kernel_constraint'][1])
        if argv['bias_constraint'] is not None:
            argv['bias_constraint'] = constraint(argv['bias_constraint'][0], argv['bias_constraint'][1])
        if argv['input_shape'] is None:
            lyr = Conv2D(filters=argv['filters'], kernel_size=argv['kernel_size'], strides=argv['strides'],
                         padding=argv['padding'], data_format=argv['data_format'], dilation_rate=argv['dilation_rate'],
                         groups=argv['groups'], activation=argv['activation'], use_bias=argv['use_bias'],
                         kernel_initializer=argv['kernel_initializer'], bias_initializer=argv['bias_initializer'],
                         kernel_regularizer=argv['kernel_regularizer'], bias_regularizer=argv['bias_regularizer'],
                         activity_regularizer=argv['activity_regularizer'], kernel_constraint=argv['kernel_constraint'],
                         bias_constraint=argv['bias_constraint'])
        else:
            lyr = Conv2D(filters=argv['filters'], kernel_size=argv['kernel_size'], strides=argv['strides'],
                         padding=argv['padding'], data_format=argv['data_format'], dilation_rate=argv['dilation_rate'],
                         groups=argv['groups'], activation=argv['activation'], use_bias=argv['use_bias'],
                         kernel_initializer=argv['kernel_initializer'], bias_initializer=argv['bias_initializer'],
                         kernel_regularizer=argv['kernel_regularizer'], bias_regularizer=argv['bias_regularizer'],
                         activity_regularizer=argv['activity_regularizer'], kernel_constraint=argv['kernel_constraint'],
                         bias_constraint=argv['bias_constraint'], input_shape=argv['input_shape'])
    elif argument['layer'] == 'MaxPooling2D':
        from keras.layers import MaxPooling2D
        argv = {'pool_size': (2, 2), 'strides': None, 'padding': 'valid', 'data_format': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = MaxPooling2D(pool_size=argv['pool_size'], strides=argv['strides'], padding=argv['padding'],
                           data_format=argv['data_format'])
    elif argument['layer'] == 'Flatten':
        from keras.layers import Flatten
        argv = {'data_format': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Flatten(data_format=argv['data_format'])
    elif 'TimeDistributed-' in argument['layer']:
        argument['layer'] = argument['layer'][16:]
        from keras.layers import TimeDistributed
        lyr = TimeDistributed(layer(argument))
    elif 'Bidirectional-' in argument['layer']:
        argument['layer'] = argument['layer'][14:]
        from keras.layers import Bidirectional
        lyr = Bidirectional(layer(argument))
    elif argument['layer'] == 'Activation':
        from keras.layers import Activation
        argv = {'Activation': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Activation(activation=argv['activation'])
    elif argument['layer'] == 'BatchNormalization':
        from keras.layers import BatchNormalization as bn
        argv = {'axis': -1, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                'beta_initializer': 'zeros',
                'gamma_initializer': 'ones', 'moving_mean_initializer': 'zeros', 'moving_variance_initializer': 'ones',
                'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None, 'gamma_constraint': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = bn(center=argv['center'], beta_initializer=argv['beta_initializer'], momentum=argv['momentum'],
                 gamma_regularizer=argv['gamma_regularizer'], gamma_initializer=argv['gamma_initializer'],
                 moving_mean_initializer=argv['moving_mean_initializer'], epsilon=argv['epsilon'], scale=argv['scale'],
                 moving_variance_initializer=argv['moving_variance_initializer'],
                 beta_regularizer=argv['beta_regularizer'],
                 beta_constraint=argv['beta_constraint'], gamma_constraint=argv['gamma_constraint'], axis=argv['axis'])
    elif argument['layer'] == 'Dropout':
        from keras.layers import Dropout
        argv = {'rate': None, 'noise_shape': None, 'seed': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Dropout(rate=argv['rate'], noise_shape=argv['noise_shape'], seed=argv['seed'])
    elif argument['layer'] == 'GaussianNoise':
        from keras.layers import GaussianNoise
        argv = {'stddev': None, 'seed': None, 'input_shape': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        if argv['input_shape'] is None:
            lyr = GaussianNoise(stddev=argv['stddev'], seed=argv['seed'])
        else:
            lyr = GaussianNoise(stddev=argv['stddev'], seed=argv['seed'], input_shape=argv['input_shape'])
    elif argument['layer'] == 'GlobalAveragePooling1D':
        from keras.layers import GlobalAveragePooling1D
        argv = {'data_format': 'channels_last', 'keepdims': False}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = GlobalAveragePooling1D(data_format=argv['data_format'], keepdims=argv['keepdims'])
    elif argument['layer'] == 'Input':
        from keras.layers import Input
        argv = {'shape': None, 'batch_size': None, 'name': None, 'dtype': None, 'sparse': None, 'tensor': None,
                'ragged': None, 'type_spec': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Input(shape=argv['shape'], batch_size=argv['batch_size'], name=argv['name'], dtype=argv['dtype'],
                    sparse=argv['sparse'], tensor=argv['tensor'], ragged=argv['ragged'], type_spec=argv['type_spec'])
    elif argument['layer'] == 'Reshape':
        from keras.layers import Reshape
        argv = {'target_shape': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Reshape(target_shape=argv['target_shape'])
    elif argument['layer'] == 'DepthwiseConv2D':
        from keras.layers import DepthwiseConv2D as dc
        argv = {'kernel_size': None, 'strides': (1, 1), 'padding': 'valid', 'depth_multiplier': 1, 'data_format': None,
                'dilation_rate': (1, 1), 'activation': None, 'use_bias': True,
                'depthwise_initializer': 'glorot_uniform',
                'bias_initializer': 'zeros', 'depthwise_regularizer': None, 'bias_regularizer': None,
                'activity_regularizer': None, 'depthwise_constraint': None, 'bias_constraint': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = dc(kernel_size=argv['kernel_size'], padding=argv['padding'], bias_constraint=argv['bias_constraint'],
                 depth_multiplier=argv['depth_multiplier'], data_format=argv['data_format'], use_bias=argv['use_bias'],
                 dilation_rate=argv['dilation_rate'], activation=argv['activation'], strides=argv['strides'],
                 depthwise_initializer=argv['depthwise_initializer'], bias_initializer=argv['bias_initializer'],
                 depthwise_regularizer=argv['depthwise_regularizer'], bias_regularizer=argv['bias_regularizer'],
                 activity_regularizer=argv['activity_regularizer'], depthwise_constraint=argv['depthwise_constraint'])
    elif argument['layer'] == 'Permute':
        from keras.layers import Permute
        argv = {'dims': None}
        for key in argument.keys():
            if key in argv.keys():
                argv[key] = argument[key]
        lyr = Permute(dims=argv['dims'])
    if base_layer is None:
        return lyr
    else:
        return lyr(base_layer)


def layers(arguments, base):
    for argument in arguments:
        base = layer(argument, base)
    return base


def layers_dict(arguments, base):
    for argument in arguments:
        base = layer_dict(argument, base)
    return base


def model_define(arguments):
    from keras.models import Sequential
    model = Sequential()
    for argument in arguments:
        model.add(layer_dict(argument))
    return model


def model_compile(model, optimizer='adam', loss=None, metrics=['accuracy']):
    """
    loss - 'mean_squared_error'
         - 'mean_squared_logarithmic_error'
         - 'mean_absolute_error'
         - 'binary_crossentropy'
         - 'hinge'
         - 'squared_hinge'
         - 'categorical_crossentropy'
         - 'sparse_categorical_crossentropy'
         - 'kullback_leibler_divergence'
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def model_fit(model, input_data=None, target_data=None, epochs=1, verbose='auto', batch_size=None, validation_data=None):
    return model.fit(input_data, target_data, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_data=validation_data)


def model_evaluate(model, x, y, verbose=1):
    loss, acc = model.evaluate(x, y, verbose=verbose)
    return loss, acc


def model_pred(model, input_data, pred_type='raw', verbose=1):
    if pred_type == 'raw':
        pred = model.predict(input_data, verbose=verbose)
    elif pred_type == 'classes':
        pred = model.predict_classes(input_data, verbose=verbose)
    elif pred_type == 'proba':
        pred = model.predict_proba(input_data, verbose=verbose)
    return pred


def load(load_type, file_name, header=None, target=None):
    if load_type in ['model', 'M', 'Model']:
        from keras.models import load_model
        model = load_model(file_name + '.h5')
        return model
    elif load_type == 'json':
        from keras.models import model_from_json
        json_file = open(file_name + '.json', 'rt')
        architecture = json_file.read()
        json_file.close()
        model = model_from_json(architecture)
        model.load_weights(file_name + '.h5')
        return model
    elif load_type in ['dataset', 'Dataset', 'data', 'Data', 'D']:
        from pandas import read_csv
        dataset = read_csv(file_name + '.csv', header=header)
        if target is not None:
            len_data = dataset.shape[0]
            dataset = dataset.iloc[:, target]
            dataset = dataset.values.reshape(len_data, 1)
        return dataset


def save(save_type, source, dfile):
    if save_type in ['model', 'M', 'Model']:
        source.save(dfile + '.h5')
    elif save_type == 'json':
        architecture = source.to_json()
        with open(dfile + '.json', 'wt') as json_file:
            json_file.write(architecture)
        source.save_weights(dfile + '.h5')
    elif save_type in ['pred', 'predictions']:
        from pandas import DataFrame
        dataframe = DataFrame({'predictions': source})
        dataframe.to_csv(dfile + '.csv')
    elif save_type == 'data':
        from pandas import DataFrame
        DataFrame(source).to_csv(dfile + '.csv')


def train_test_split(input_data, target_data, test_size=0.25, split_point=None, shuffle=False):
    if split_point is None:
        x_train, x_test = input_data[:split_point, :], input_data[split_point:, :]
        y_train, y_test = target_data[:split_point], target_data[split_point:]
    else:
        from sklearn.model_selection import train_test_split as split
        x_train, x_test, y_train, y_test = split(input_data, target_data, test_size=test_size, shuffle=shuffle)
    return x_train, x_test, y_train, y_test


def optimizer(optimizer_type='SGD', learning_rate=0.01, momentum=0.):
    from keras.optimizer_v1 import SGD, RMSprop, Adam, Adagrad
    opt = {
        'SGD': SGD,
        'RMSprop': RMSprop,
        'Adam': Adam,
        'Adagrad': Adagrad
    }
    return opt[optimizer_type](learning_rate=learning_rate, momentum=momentum)


def regularizer(reg_type, value):
    from keras.regularizers import l1, l2, l1_l2
    reg = {'l1': l1, 'l2': l2, 'l1_l2': l1_l2}
    return reg[reg_type](value)


def constraint(cons_type, value):
    from keras.constraints import max_norm, unit_norm
    cons = {'max_norm': max_norm, 'init_norm': unit_norm}
    return cons[cons_type](value)


def callbacks(argument):
    if argument[0] == 'EarlyStopping':
        from keras.callbacks import EarlyStopping
        return EarlyStopping(monitor=argument[1], mode=argument[2], verbose=argument[3])
    if argument[0] == 'ModelCheckpoint':
        from keras.callbacks import ModelCheckpoint
        return ModelCheckpoint(filepath=argument[1], monitor=argument[2], mode=argument[3], verbose=argument[4],
                               save_best_only=argument[5])


def model_clone(model):
    from keras.models import clone_model
    return clone_model(model)


def ensemble_input_output(members):
    return [model.input for model in members], [model.output for model in members]


def concatenate(lyrs):
    from keras.layers.merge import concatenate
    return concatenate(lyrs)


def model_Model(inputs, outputs):
    from keras.models import Model
    return Model(inputs=inputs, outputs=outputs)


def makedirs(dname):
    from os import makedirs as md
    md(dname)


def define_stacked_model(members, hidden, output, loss, plot_graph=False):
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    ensemble_visible = [model.input for model in members]
    ensemle_outputs = [model.output for model in members]
    merge = concatenate(ensemle_outputs)
    hidden = hidden(merge)
    output = output(hidden)
    model = model_Model(ensemble_visible, output)
    if plot_graph:
        from tensorflow.keras.utils import plot_model
        plot_model(model, show_shapes=True, to_file='model_graph.png')
    model = model_compile(model, loss=loss)
    return model


def fit_stacked_model(model, input_x, input_y, epochs=1, verbose='auto'):
    from tensorflow.keras.utils import to_categorical
    x = [input_x for _ in range(len(model.input))]
    input_y_enc = to_categorical(input_y)
    model = model_fit(model, x, input_y_enc, epochs=epochs, verbose=verbose)
    return model


def predict_stacked_model(model, input_x):
    x = [input_x for _ in range(len(model.input))]
    return model_pred(model, x, verbose=0)


def load_all_models(n_start, n_end, fname):
  all_models = list()
  for epoch in range(n_start, n_end+1):
    # define filename for this ensemble
    filename = fname + str(epoch)
    # load model from file
    model = load('model', filename)
    # add to list of members
    all_models.append(model)
  return all_models


def model_weight_ensemble(members, weights):
  from numpy import array, average
  # determine how many layers need to be averaged
  n_layers = len(members[0].get_weights())
  # create an set of average model weights
  avg_model_weights = list()
  for layer in range(n_layers):
    # collect this layer from each model
    layer_weights = array([model.get_weights()[layer] for model in members])
    # weighted average of weights for this layer
    avg_layer_weights = average(layer_weights, axis=0, weights=weights)
    # store average layer weights
    avg_model_weights.append(avg_layer_weights)
  # create a new model with the same structure
  model = model_clone(members[0])
  # set the weights in the new
  model.set_weights(avg_model_weights)
  model_compile(model, loss='categorical_crossentropy')
  return model


def evaluate_n_members(members, n_members, testX, testy):
  # select a subset of members
  subset = members[:n_members]
  # prepare an array of equal weights
  weights = [1.0/n_members for i in range(1, n_members+1)]
  # create a new model with the weighted average of all model weights
  model = model_weight_ensemble(subset, weights)
  model_compile(model)
  # make predictions and evaluate accuracy
  _, test_acc = model_evaluate(model, testX, testy, verbose=0)
  return test_acc


def classification_lstm_main(argv_prepare, argv_model, argv_pred, argv_eval, argv_save=[False], argv_plot=[False]):
    from time import time
    start_time = time()

    # Step 1 --> data preparation
    if argv_prepare[0] == 'generate':
        # x, y = generate_dataset(n_samples=1000, n_features=4, generate_type='regression', noise=0.2, random_state=5)
        x, y = generate_dataset(n_samples=argv_prepare[1], n_features=argv_prepare[2], generate_type=argv_prepare[3],
                                noise=argv_prepare[4], random_state=argv_prepare[5])
    elif argv_prepare[0] == 'load':
        dataset = load(load_type='dataset', file_name=argv_prepare[1])
        x = dataset.iloc[:, dataset.columns != argv_prepare[2]].values
        len_data = dataset.shape[0]
        y = dataset.iloc[:, argv_prepare[2]].values
        y = y.reshape(len_data, 1)

    # Step 2 --> train a model
    if argv_model[0] == 'train':
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=argv_model[1])
        argv_model[2]['shape'] = (x_train.shape[0], x_train.shape[1])
        ip = layer_dict(argv_model[2])
        x = layers_dict(argv_model[3], ip)
        y = layers_dict(argv_model[4], ip)
        y = concatenate([x, y])

        y_layers = list()
        for i in range(argv_model[5]):
            y_layers.appennd(layer_dict(argv_model[6], y))
        y = concatenate(y_layers)
        z = layers_dict(argv_model[7], y)
        model = model_Model(ip, z)

        model = model_compile(model, optimizer=optimizer(argv_model[8]), loss=argv_model[9], metrics=argv_model[10])
        # model = model_compile(model, optimizer=optimizer('Adam'), loss='mean_squared_error', metrics=['accuracy'])

        model = model_fit(model, input_data=x_train, target_data=y_train, epochs=argv_model[11], batch_size=argv_model[12])
        # model = model_fit(model, input_data=x_train, target_data=y_train, epochs=50, batch_size=4)
        if argv_model[13]:
            save('model', model, argv_model[13])
    elif argv_model[0] == 'load':
        x_test, y_test = x, y
        model = load(load_type='model', file_name=argv_model[1])

    # Step 3 --> predict
    pred = model_pred(model, x_test, verbose=argv_pred[0])

    # Step 4 --> evaluate results
    from source_common import evaluate_error
    error = evaluate_error(y_test, pred, metrics=argv_eval[0])
    exec_time = time() - start_time
    print('{RMSE: %.3f, execution_time: %.3f}' % (error, exec_time))

    # Step 5 --> save results
    if argv_save[0]:
        save(save_type='pred', source=pred, dfile=argv_save[0])

    # Step 6 --> plot
    if argv_plot[0]:
        from source_common import plotting
        plotting([y_test, pred], method='L')


def lstm_main(argv_prepare, argv_model, argv_pred, argv_evaluate, argv_save_pred, argv_plot):
    """
    argv_prepare[0] - file_name /str/ in .csv file format
    argv_prepare[1] - target /int/ None or any columns index for targeting
    argv_prepare[2] - do_scale /boolean/
    argv_prepare[3] - scaler_type /str/ 'MinMaxScaler' or 'StandardScaler' *if do_scale is True
    argv_prepare[4] - feature_range /tuple/ None or range for features
    ------------------------------------------------------------------
    argv_model[0] - model_name /str/
    argv_model[1] - save_load_type /str/ 'model' or 'json'
    argv_model[2] - do_train /boolean/
    argv_model[3] - look_back /int/
    argv_model[4] - forward_days /int/
    argv_model[5] - periods /int/
    argv_model[6] - argument /list/ - contains several layers' arguments
    e.g. argument = [['LSTM', ...], ['Dense', ...]]
         argument[0] = ['LSTM', ...] - adds LSTM layer to the model
         argument[1] = ['Dense', ...] - adds Dense layer to the model
    argv_model[7] - optimizer /str/ 'rmsprop', 'adam' or 'adagrad'
    argv_model[8] - loss /str/ 'mean_squared_error', 'mae' or ...
    argv_model[9] - metrics /str/ None, or ...
    argv_model[10] - epochs /int/
    argv_model[11] - verbose /int/
    argv_model[12] - batch_size /int/
    ------------------------------------------------------------------
    argv_pred[0] - pred_type /str/ 'raw', 'classes' or 'proba'
    ------------------------------------------------------------------
    argv_evaluate[0] - metrics /str/ 'MAE', 'MSE' or 'RMSE'
    ------------------------------------------------------------------
    argv_save_pred[0] - do_save /boolean/
    argv_save_pred[1] - dfile /str/ destination file name
    ------------------------------------------------------------------
    argv_plot[0] - do_plot /boolean/
    argv_plot[0] - method /str/ 'L', 'D.', 'H', 'De', 'S' or ...
    ------------------------------------------------------------------
    """

    from time import time
    start_time = time()

    # Step 1 --> data preparation
    if argv_prepare[0] == 'generate':
        # x, y = generate_dataset(n_samples=1000, n_features=4, generate_type='regression', noise=0.2, random_state=5)
        x, y = generate_dataset(n_samples=argv_prepare[1], n_features=argv_prepare[2], generate_type=argv_prepare[3],
                                noise=argv_prepare[4], random_state=argv_prepare[5])
    elif argv_prepare[0] == 'load':
        dataset = load(load_type='dataset', file_name=argv_prepare[1])
        x = dataset.iloc[:, dataset.columns != argv_prepare[2]].values
        len_data = dataset.shape[0]
        y = dataset.iloc[:, argv_prepare[2]].values
        y = y.reshape(len_data, 1)

    # Step 2 --> model
    if argv_model[0] == 'train':  # train model
        # split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=argv_model[1])

        # define and compile model
        model = model_define(arguments=argv_model[2])
        model = model_compile(model=model, optimizer=argv_model[3], loss=argv_model[4], metrics=argv_model[5])

        # fit model
        model = model_fit(model=model, input_data=x_train, target_data=y_train,
                          epochs=argv_model[6], verbose=argv_model[7], batch_size=argv_model[8])

        # save model
        save(save_type='model', source=model, dfile=argv_model[9])
    elif argv_model[0] == 'load':
        x_test, y_test = x. y
        # load model
        model = load(load_type=argv_model[1], file_name=argv_model[0])

    # Step 3 --> predict
    pred = model_pred(pred_type=argv_pred[0], model=model, input_data=x_test)

    # Step 4 --> evaluate error
    from source_common import evaluate_error
    error = evaluate_error(y_test, pred, metrics=argv_evaluate[0])
    exec_time = time() - start_time
    print('{%s: %.3f, execution_time: %.3f}' % (argv_evaluate[0], error, exec_time))

    # Step 5 --> save results
    if argv_save_pred[0]:
        save(save_type='pred', source=pred, dfile=argv_save_pred[0])

    # Step 6 --> plot
    if argv_plot[0]:
        from source_common import plotting
        plotting([y_test, pred], method='L')


def ensamble_model(load_members, x_test, y_test, do_print=False, do_plot=False):
    members = load_all_models(n_start=load_members[0], n_end=load_members[1], fname=load_members[2])
    if load_members[3]:
        members = list(reversed(members))
    single_scores, ensamble_scores = list(), list()
    for i in range(1, len(members)+1):
        ensamble_score = evaluate_n_members(members, i, x_test, y_test)
        _, single_score = model_evaluate(model=members[i-1], x=x_test, y=y_test, verbose=0)
        if do_print:
            print('> %d: single: %.3f, ensamble: %.3f' % (i, single_score, ensamble_score))
        single_scores.append(single_score)
        ensamble_scores.append(ensamble_score)
    if do_plot:
        from source_common import plotting
        plotting([single_scores, ensamble_scores], 'Line')


def stacked_model(load_members, x_test, y_test, hidden, output, do_print=False):
    from numpy import argmax
    from sklearn.metrics import accuracy_score
    members = load_all_models(n_start=load_members[0], n_end=load_members[1], fname=load_members[2])
    print('loaded members: ', len(members))
    single_scores = list()
    for i in range(len(members)):
        _, single_score = model_evaluate(model=members[i], x=x_test, y=y_test, verbose=0)
        if do_print:
            print('> %d: single: %.3f' % (i+1, single_score))
        single_scores.append(single_score)
    model = define_stacked_model(members, hidden=hidden, output=output, loss='categorical_crossentropy', plot_graph=True)
    fit_stacked_model(model, x_test, y_test)
    yhat = predict_stacked_model(fit_model, x_test)
    yhat = argmax(yhat, axis=1)
    acc = accuracy_score(test, yhat)
    print('Stacked Test Accuracy: %.3f' % acc)


# make an ensemble prediction fo mul-class classification
def ensemble_predictions(members, weights, test_x):
    from numpy import array, tensordot, argmax
    # make predictions
    yhats = [model.predict(test_x) for model in members]
    yhats = array(yhats)
    # weighted sum across classes
    summed = tensordot(yhats, weights, axes=((0), (0)))
    # argmax across classes
    result = argmax(summed, axis=1)
    return result


# evaluate a specific number of members in an ensemble
def ensemble_evaluate(members, weights, test_x, test_y):
    from sklearn.metrics import accuracy_score
    # make predictions
    yhat = ensemble_predictions(members, weights, test_x)
    # calculate accuracy
    acc = accuracy_score(test_y, yhat)
    return acc


def weight_normalize(weights):
    from numpy.linalg import norm
    # calculate l1 vector norm
    result = norm(weights)
    if result == 0.0:
        return weights
    return weights / result


def loss_function(weights, members, test_x, test_y):
    normalized_weights = weight_normalize(weights)
    return 1.0 - ensemble_evaluate(members, normalized_weights, test_x, test_y)


def client_train(x_s, y_s, arg_model, epochs, dfilename, verbose=0):
    for i in range(len(x_s)):
        len_ = x_s[i].shape[0]
        x_train = x_s[i][int(len_ * 0.3):]
        y_train = y_s[i][int(len_ * 0.3):]
        x_val = x_s[i][:int(len_ * 0.3)]
        y_val = y_s[i][:int(len_ * 0.3)]
        model = model_define(arg_model)
        model_compile(model, loss='categorical_crossentropy')
        model_fit(model, x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), verbose=verbose)
        save('model', model, f'{dfilename}{i+1}')


def server_ensemble(members, ens_type, test_x=None, test_y=None, maxiter=1000, tol=0.01, weights=None):
    n_members = len(members)
    if ens_type == 'averaging':
        weights = [1.0/n_members for _ in range(n_members)]
        ens_model = model_weight_ensemble(members, weights)
    elif ens_type == 'optimized':
        from scipy.optimize import differential_evolution
        weights = [(0.0, 1.0) for _ in range(n_members)]
        args = (members, test_x, test_y)
        result = differential_evolution(loss_function, weights, args, maxiter=maxiter, tol=tol)
        optimized_weights = weight_normalize(result['x'])
        ens_model = model_weight_ensemble(members, optimized_weights)
    elif ens_type == 'best_case':
        from tensorflow.keras.utils import to_categorical
        test_y_enc = to_categorical(test_y)
        best_score, best_model = 0.0, None
        for model in members:
            _, acc = model_evaluate(model, test_x, test_y_enc)
            if acc > best_score:
                best_model = model
                best_score = acc
        ens_model = best_model
    elif ens_type == 'custom_weight':
        ens_model = model_weight_ensemble(members, weights)
    save('model', ens_model, 'ens_model')


def client_test(members, ens_model, test_x, test_y, dfilename='test_results'):
    from tensorflow.keras.utils import to_categorical
    test_y_enc = to_categorical(test_y)
    results = list()
    # 1 result is result of ensemble model
    _, acc = model_evaluate(ens_model, test_x, test_y_enc)
    results.append(acc)
    # rest of results for member model
    for model in members:
        _, acc = model_evaluate(model, test_x, test_y_enc)
        results.append(acc)
    # save results
    save('data', results, dfilename)


def result_comparision(filename):
    results = list(load('data', filename, header=0).iloc[:, 1].values)
    ens_model_result = results[0]
    members_results = results[1:]
    print(f'>ens_model: {ens_model_result:.3f}')
    for i in range(len(members_results)):
        print(f'>model {i+1:2d}: {members_results[i]:.3f}')
