from tensorflow.keras import Sequential, Model
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


class ActorCriticModel(Model):
    def __init__(self, nb_actions=4):
        super(ActorCriticModel, self).__init__()
        self.nb_actions = nb_actions

        self.flatten = Flatten()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.fc3 = Dense(256, activation='relu')
        self.fc4 = Dense(256, activation='relu')
        self.fc5 = Dense(256, activation='relu')
        self.v = Dense(1)
        self.pi = Dense(self.nb_actions, activation='softmax')

    def call(self, state, *args):
        x = self.flatten(state)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        v = self.v(x)
        pi = self.pi(x)
        return v, pi


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


def get_policy_network(nb_actions=4, dnn_name='dnn3', mode='pg'):
    if dnn_name == 'dnn3':
        model = dnn3(nb_actions=nb_actions)
    elif dnn_name == 'dnn5':
        if mode == 'pg':
            model = dnn5(nb_actions=nb_actions)
        elif mode == 'a2c':
            model = ActorCriticModel()
    elif dnn_name == 'cnn':
        model = cnn(nb_actions=nb_actions)
    elif dnn_name == 'cnnd':
        model = cnnd(nb_actions=nb_actions)
    elif dnn_name == 'cnndv2':
        model = cnndv2(nb_actions=nb_actions)
    elif dnn_name == 'minidnn':
        model = minidnn(nb_actions=nb_actions)

    return model
