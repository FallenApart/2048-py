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
        return actions.numpy()

    def learn(self, batch_state_memory, batch_action_memory, batch_reward_memory):
        batch_G = []
        for reward_memory in batch_reward_memory:
            G = np.zeros_like(reward_memory)
            G_tmp = 0
            for t in reversed(range(len(reward_memory))):
                G[t] = G_tmp * self.gamma + reward_memory[t]
                G_tmp = G[t]
            batch_G.append(G)

        with tf.GradientTape() as tape:
            losses = []
            for state_memory, action_memory, G in zip(batch_state_memory, batch_action_memory, batch_G):
                action_probs = self.get_action_probs(state_memory, training=True)
                log_probs = action_probs.log_prob(action_memory)
                loss = - tf.reduce_sum(log_probs * G)
                losses.append(loss)
            loss = tf.reduce_mean(losses)

            if tf.math.is_nan(loss):
                print('loss is nan')
            else:
                grads = tape.gradient(loss, self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
