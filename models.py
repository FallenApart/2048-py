from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, GlobalAvgPool2D, Softmax, Input


def minidnn(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model


def dnn3(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Flatten(),
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model


def dnn5(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Flatten(),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model


def cnn(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Conv2D(128, (4, 4), padding='same', activation='relu'),
                        Conv2D(256, (4, 4), padding='same', activation='relu'),
                        Conv2D(512, (4, 4), padding='same', activation='relu'),
                        Conv2D(256, (4, 4), padding='same', activation='relu'),
                        Conv2D(64, (4, 4), padding='same', activation='relu'),
                        Conv2D(nb_actions, (4, 4), padding='same'),
                        GlobalAvgPool2D(),
                        Softmax()])
    return model


def cnnd(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Conv2D(128, (4, 4), padding='same', activation='relu'),
                        Conv2D(256, (4, 4), padding='same', activation='relu'),
                        GlobalAvgPool2D(),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model


def cnndv2(nb_actions=4):
    model = Sequential([Input(shape=(4, 4, 1)),
                        Conv2D(512, (4, 4), padding='valid', activation='relu'),
                        GlobalAvgPool2D(),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model


def get_policy_network(nb_actions=4, dnn_name='dnn3'):
    if dnn_name == 'dnn3':
        model = dnn3(nb_actions=nb_actions)
    elif dnn_name == 'dnn5':
        model = dnn5(nb_actions=nb_actions)
    elif dnn_name == 'cnn':
        model = cnn(nb_actions=nb_actions)
    elif dnn_name == 'cnnd':
        model = cnnd(nb_actions=nb_actions)
    elif dnn_name == 'cnndv2':
        model = cnndv2(nb_actions=nb_actions)
    elif dnn_name == 'minidnn':
        model = minidnn(nb_actions=nb_actions)

    return model
