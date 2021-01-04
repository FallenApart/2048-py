import time
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import get_policy_network
from abc_agent import ABCAgent


class DNNAgent(ABCAgent):
    def __init__(self, lr=0.0005, gamma=0.99, nb_actions=4, dnn_name='dnn3'):
        super(DNNAgent, self).__init__(nb_actions=nb_actions)
        self.lr = lr
        self.gamma = gamma
        self.dnn_name = dnn_name
        self.policy = get_policy_network(self.nb_actions, self.dnn_name)
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
        return actions.numpy()[0]

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

        self.reset_memory()

        return feedforward_time, backprop_time
