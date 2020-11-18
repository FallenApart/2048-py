from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten


def get_policy_network(nb_actions=4):
    model = Sequential([Flatten(),
                        Dense(256, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(nb_actions, activation='softmax')])
    return model
