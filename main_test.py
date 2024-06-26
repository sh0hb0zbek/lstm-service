from source_lstm import *
from source_common import *


def test_classification_lstm_main():

    argv_prepare = ['generate', 1000, 4, 'regression', 0.2, 5]
    argv_model = [
        'train',
        0.1,
        {'layer': 'Input'},
        [{'layer': 'Permute', 'dims': (2, 1)},
         {'layer': 'LSTM', 'units': 10},
         {'layer': 'Dropout', 'rate': 0.8}],
        [{'layer': 'Reshape', 'target_shape': (128, 1, 9)},
         {'layer': 'DepthwiseConv2D', 'depth_multiplier': 20, 'data_format': 'channels_last', 'padding': 'valid'},
         {'layer': 'Activation', 'activation': 'relu'},
         {'layer': 'MaxPooling2D', 'pool_size': (5, 1), 'strides': (1, 2), 'padding': 'valid'},
         {'layer': 'DepthwiseConv2D', 'kernel_size': (7, 1), 'depth_multiplier': 10, 'data_format': 'channels_last', 'padding': 'valid'},
         {'layer': 'Activation', 'activation': 'relu'},
         {'layer': 'Reshape', 'target_shape': (110, 1800)},
         {'layer': 'GlobalAveragePooling1D'}],
        32,
        {'layer': 'Dense', 'units': 1, 'actication': 'relu'},
        [{'layer': 'Dropout'},
         {'layer': 'Dense', 'units': 6, 'activation': 'softmax'}],
        'Adam',
        'mean_squared_error',
        ['accuracy'],
        50,
        4,
        'model'
    ]
    argv_pred = [2]
    argv_eval = ['RMSE']
    argv_save = ['pred']
    argv_plot = [True]
    from source_lstm import classification_lstm_main
    classification_lstm_main(argv_prepare, argv_model, argv_pred, argv_eval, argv_save, argv_plot)


def test_lstm_main():
    prepare = ['generate', 500, 4, 'regression', 0.2, 3]
    pred = ['raw']
    evaluate = ['RMSE']
    save_pred = [False]
    plot = [True, 'L']
    vanilla_model = [
        'train', 0.1,
        [
            {'layer': 'LSTM', 'units': 25, 'unit_shape': (5, 10)},
            {'layer': 'Dense', 'units': 10, 'activation': 'softmax'}
        ],
        'adam', 'categorical_crossentropy', None, 1, 2, 4
    ]
    stacked_model = [
        'train', 0.1,
        [
            {'layer': 'LSTM', 'units': 20, 'return_sequences': True, 'unit_shape': (50, 1)},
            {'layer': 'LSTM', 'units': 20},
            {'layer': 'Dense', 'units': 5}
        ],
        'adam', 'mae', None, 1, 1, 10
    ]
    cnn_model = [
        'train', 0.1,
        [
            {'layer': 'TimeDistributed-Conv2D', 'filters': 2, 'kernel_size': (2, 2), 'activation': 'relu', 'input_shape': (None, 50, 50, 1)},
            {'layer': 'TimeDistributed-MaxPooling2D', 'pool_size': (2, 2)},
            {'layer': 'TimeDistributed-Flatten'},
            {'layer': 'LSTM', 'units': 50},
            {'layer': 'Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'adam', 'binary_crossentropy', None, 1, 1, 32
    ]
    encoder_decoder_model = [
        'train', 0.1,
        [
            {'layer': 'LSTM', 'units': 75, 'unit_shape': (8, 12)},
            {'layer': 'RepeatVector', 'n': 2},
            {'layer': 'LSTM', 'units': 50, 'return_sequences': True},
            {'layer': 'TimeDistributed-Dense', 'units': 12, 'activation': 'softmax'}
        ],
        'adam', 'categorical_crossentropy', None, 1, 1, 32
    ]
    bidirectional_model = [
        'train', 0.1,
        [
            {'layer': 'Bidirectional-LSTM', 'units': 50, 'return_sequences': True, 'input_shape': (10, 1)},
            {'layer': 'TimeDistributed-Dense', 'units': 1, 'activation': 'sigmoid'}
        ],
        'adam', 'binary_crossentropy', ['accuracy'], 1, 1, 10
    ]
    generative_model = [
        'train', 0.1,
        [
            {'layer': 'LSTM', 'units': 10, 'return_sequences': False, 'input_shape': (1, 2)},
            {'layer': 'Dense', 'units': 2, 'actovation': 'linear'}
        ],
        'adam', 'mae', None, 1, 2, 16
    ]

    models = [
        [prepare, vanilla_model, pred, evaluate, save_pred, plot],
        [prepare, stacked_model, pred, evaluate, save_pred, plot],
        [prepare, cnn_model, pred, evaluate, save_pred, plot],
        [prepare, encoder_decoder_model, pred, evaluate, save_pred, plot],
        [prepare, bidirectional_model, pred, evaluate, save_pred, plot],
        [prepare, generative_model, pred, evaluate, save_pred, plot],
    ]
    for model in models:
        lstm_main(model[0], model[1], model[2], model[3], model[4], model[5])


def test_transform_n_scaler():
    from pandas import Series
    # define contrived series
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    series = Series(data)
    print(series)
    # prepare data for normalization
    values = series.values
    values = values.reshape((len(values), 1))
    # train the normalization
    scl = scaler(series=values, scaler_type='MMS', feature_range=(0, 1))
    # normalize the dataset and print
    normalized = normalizer(scl, values=values)
    print(normalized)
    # inverse transform and print
    inversed = normalizer(scl, values, inverse=True)
    print(inversed)


def test_encode():
    from numpy import array
    # define example
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
    values = array(data)
    # integer encode - LabelEncoder
    label_encoded = encode(values=values, method='LE')
    print(label_encoded)
    # binary encode - OneHotEncoder
    one_hot_encoded = encode(values=label_encoded, method='OHE', sparse=False, categories='auto')
    print(one_hot_encoded)


def test_sequence_padding():
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3],
        [1]
    ]
    # pre-sequence padding
    pre_padded = padding(sequences, method='pre')
    print(pre_padded)
    # post_sequence padding
    post_padded = padding(sequences, method='post')
    print(post_padded)
    # pre-sequence truncation
    pre_truncation = padding(sequences, method='pre', truncation=True, maxlen=2)
    print(pre_truncation)
    #post-sequence truncation
    post_truncation = padding(sequences, method='post', truncation=True, maxlen=2)
    print(post_truncation)


def test_generate_dataset():
    sequence = generate_dataset(n_samples=100, n_features=6, generate_type='blobs')
    print(sequence)


def test_model_clone():
    x, y = generate_dataset(n_samples=100, n_features=2, generate_type='regression')
    model = model_define([['LSTM', 15, False, None, None, None, None, None, None, None]])
    model = model_compile(model, optimizer('Adam'), loss='mae', metrics=['accuracy'])
    model = model_fit(model, x, y, epochs=1, batch_size=4)

    cloned = model_clone(model)
    print('summary of model\n', model.summary())
    print('summary of cloned model\n', cloned.summary())


def test_ensemble_model():
    from tensorflow.keras.utils import to_categorical
    x, y = generate_dataset(1100, 2, 'blobs', centers=3, cluster_std=2, random_state=2)
    y = to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, split_point=100)
    model = model_define([
        {'layer': 'Dense', 'units': 25, 'input_dim': 2, 'actovation': 'relu'},
        {'layer': 'Dense', 'units': 3, 'activation': 'softmax'}
    ])
    model_compile(model, loss='categorical_crossentropy')
    for i in range(10):
        model_fit(model, x_train, y_train, epochs=1, verbose=0, validation_data=(x_test, y_test))
        save('model', source=model, dfile='model_'+str(i+1))
    ensamble_model([1, 11, 'model_', True], x_test, y_test, True, True)


def test_stacked_model():
    from tensorflow.keras.utils import to_categorical
    x, y = generate_dataset(1100, 2, 'blobs', centers=3, cluster_std=2, random_state=2)
    y = to_categorical(y)
    n_train=100
    x_train, x_test, y_train, y_test = train_test_split(x, y, split_point=n_train)
    model = model_define([
        {'layer': 'Dense', 'units': 25, 'unit_dim': 2, 'activation': 'relu'},
        {'layer': 'Dense', 'units': 3, 'activation': 'softmax'}
    ])
    model_compile(model, loss='categorical_crossentropy')
    for i in range(10):
        model_fit(model, x_train, y_train, epochs=1, verbose=0)
        save('model', model, 'model_'+str(i+1))
    hidden = layer_dict({'layer': 'Dense', 'units': 10, 'activation': 'relu'})
    output = layer_dict({'layer': 'Dense', 'units': 3, 'activation': 'softmax'})
    stacked_model([1, 11, 'model_'], x_test, y_test, hidden, output, True)


def distributed_ensemble_test():
    from time import time
    start = time()
    # preparation --> make dataset x, y
    x, y = generate_dataset(n_samples=100000, n_features=3, generate_type='blobs', centers=3, cluster_std=2,
                            random_state=2)
    # x_train, y_train (70%) will splitted to 4 parts to train member models
    # test_x, test_y (30%) will be divided into two parts for evaluation (24%) and ensemble usage (6%)
    x_train, test_x, y_train, test_y = train_test_split(x, y, test_size=0.3)
    test_x, ens_test_x, test_y, ens_test_y = train_test_split(test_x, test_y, test_size=0.2)

    len_train = x_train.shape[0]
    from tensorflow.keras.utils import to_categorical
    # 10% of train data is for model1
    train_x_model1 = x_train[:int(0.1 * len_train)]
    train_y_model1 = to_categorical(y_train[:int(0.1 * len_train)])
    # 10% of train data is for model2
    train_x_model2 = x_train[int(0.1 * len_train):int(0.3 * len_train)]
    train_y_model2 = to_categorical(y_train[int(0.1 * len_train):int(0.3 * len_train)])
    # 10% of train data is for model3
    train_x_model3 = x_train[int(0.3 * len_train):int(0.6 * len_train)]
    train_y_model3 = to_categorical(y_train[int(0.3 * len_train):int(0.6 * len_train)])
    # 10% of train data is for model4
    train_x_model4 = x_train[int(0.6 * len_train):]
    train_y_model4 = to_categorical(y_train[int(0.6 * len_train):])

    x_s, y_s = list(), list()
    x_s.append(train_x_model1)
    x_s.append(train_x_model2)
    x_s.append(train_x_model3)
    x_s.append(train_x_model4)
    y_s.append(train_y_model1)
    y_s.append(train_y_model2)
    y_s.append(train_y_model3)
    y_s.append(train_y_model4)

    time_preparation = time() - start

    start = time()

    # ---------> step 1 --> client will train models based on training datasets
    arg_model = [{'layer': 'Dense', 'units': 25, 'unit_dim': 2, 'activation': 'relu'},
                 {'layer': 'Dense', 'units': 3, 'activation': 'softmax'}]
    epochs = 10
    # makedirs('models')
    dfilename = 'models/model_'
    client_train(x_s, y_s, arg_model, epochs, dfilename)

    time_train_save = time() - start

    # load members
    members = load_all_models(1, 4, dfilename)

    start = time()

    # ---------> step 2 --> make&save ensemble model
    ens_type = 'optimized'
    weights = None
    server_ensemble(members, ens_type, test_x=ens_test_x, test_y=ens_test_y, weights=weights)

    # load ensemble model
    ens_model = load('model', 'ens_model')

    time_ensemble = time() - start

    start = time()

    # ---------> step 3 --> test member models and ensemble model by using test_x&test_y (24% of dataset)
    #                   --> save results to the file results.csv
    dfilename = 'results'
    client_test(members, ens_model, test_x, test_y, 'results')

    time_test = time() - start

    # ---------> step 4 --> print results
    result_comparision('results')

    print("{'time for prepagation': %.3f,\n"
          "'time foe training and saving': %.3f,\n"
          "'time for ensemble model': %.3f,\n"
          "'time for test': %3.f}" % (time_preparation, time_train_save, time_ensemble, time_test))


if __name__ == '__main__':
    distributed_ensemble_test()
    # test_stacked_model()
    # test_ensemble_model()
    # test_classification_lstm_main()
    # test_lstm_main()
    # test_transform_n_scaler()
    # test_encode()
    # test_sequence_padding()
    # test_generate_dataset()
    # test_model_clone()
