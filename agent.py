import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import get_policy_network


class Agent:
    def __init__(self, lr=0.0005, gamma=0.99, nb_actions=4, name='dnn3'):
        self.lr = lr
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.valid_actions_memory = []
        self.policy = get_policy_network(self.nb_actions, name)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        self.policy.summary()

    def get_action_probs(self, states, training=None):
        probs = self.policy(np.expand_dims(states, axis=-1), training=training)
        action_probs = tfp.distributions.Categorical(probs=probs)
        return action_probs

    def choose_actions(self, states):
        with tf.device("/cpu:0"):
            actions_probs = self.get_action_probs(states, training=False)
            actions = actions_probs.sample()
        return actions

    def store_transition(self, state_np, action_np, reward_np):
        self.state_memory.append(state_np)
        self.action_memory.append(action_np)
        self.reward_memory.append(reward_np)

    def learn(self):
        G = np.zeros_like(self.reward_memory)
        G_tmp = 0
        for t in reversed(range(len(self.reward_memory))):
            G[t] = G_tmp * self.gamma + self.reward_memory[t]
            G_tmp = G[t]

        start_time = time.time()

        with tf.GradientTape() as tape:

            action_probs = self.get_action_probs(np.array(self.state_memory), training=True)
            log_probs = action_probs.log_prob(np.array(self.action_memory))
            loss = - tf.reduce_sum(G * log_probs)

            feedforward_time = time.time() - start_time
            start_time = time.time()

            if tf.math.is_nan(loss):
                print('loss is nan')
            else:
                grads = tape.gradient(loss, self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

            backprop_time = time.time() - start_time

        self.state_memory, self.action_memory, self.reward_memory = [], [], []

        return feedforward_time, backprop_time
