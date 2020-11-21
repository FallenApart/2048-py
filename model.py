from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout


def get_policy_network(nb_actions=4):
    model = Sequential([Flatten(),
                        Dense(256, activation='relu'),
                        Dropout(0.1),
                        Dense(256, activation='relu'),
                        Dropout(0.1),
                        Dense(256, activation='relu'),
                        Dropout(0.1),
                        Dense(nb_actions, activation='softmax')])
    return model
