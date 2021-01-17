import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import get_policy_network
from abc_agent import ABCAgent


class DNNAgent(ABCAgent):
    def __init__(self, mode='pg', lr=0.0005, gamma=0.99, nb_actions=4, dnn_name='dnn3'):
        super(DNNAgent, self).__init__(nb_actions=nb_actions)
        self.mode = mode
        self.lr = lr
        self.gamma = gamma
        self.dnn_name = dnn_name
        self.policy = get_policy_network(self.nb_actions, self.dnn_name, self.mode)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))
        # self.policy.summary()

    def get_action_probs(self, states, training=None):
        if self.mode == 'pg':
            probs = self.policy(states, training=training)
        elif self.mode == 'a2c':
            _, probs = self.policy(states, training=training)
        action_probs = tfp.distributions.Categorical(probs=probs)
        return action_probs

    def choose_actions(self, states):
        with tf.device("/cpu:0"):
            actions_probs = self.get_action_probs(states, training=False)
            actions = actions_probs.sample()
        return actions.numpy()

    def learn(self, batch_state_memory, batch_action_memory, batch_reward_memory, batch_terminal_state):
        if self.mode == 'pg':
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
            if self.mode == 'pg':
                for state_memory, action_memory, G in zip(batch_state_memory, batch_action_memory, batch_G):
                    action_probs = self.get_action_probs(state_memory, training=True)
                    log_probs = action_probs.log_prob(action_memory)
                    loss = - tf.reduce_sum(log_probs * G)
                    losses.append(loss)
            elif self.mode == 'a2c':
                for state_memory, action_memory, reward_memory, terminal_state in zip(batch_state_memory,
                                                                                      batch_action_memory,
                                                                                      batch_reward_memory,
                                                                                      batch_terminal_state):
                    state_value, probs = self.policy(np.array(state_memory), training=True)
                    next_state_value, _ = self.policy(np.array(state_memory[1:] + [terminal_state]), training=True)
                    action_probs = tfp.distributions.Categorical(probs=probs)
                    log_probs = action_probs.log_prob(np.array(action_memory))
                    dones = np.ones((next_state_value.shape[0],))
                    dones[-1] = 0
                    deltas = np.array(reward_memory) + self.gamma * next_state_value * dones - state_value
                    actor_loss = - tf.reduce_sum(log_probs * deltas)
                    critic_loss = tf.reduce_sum(deltas ** 2)
                    loss = actor_loss + critic_loss
                    losses.append(loss)
            loss = tf.reduce_mean(losses)

            if tf.math.is_nan(loss):
                print('loss is nan')
            else:
                grads = tape.gradient(loss, self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
