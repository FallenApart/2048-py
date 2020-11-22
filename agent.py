import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np
from model import get_policy_network


class Agent:
    def __init__(self, lr=0.0005, gamma=0.99, nb_actions=4):
        self.lr = lr
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.policy = get_policy_network(self.nb_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    def get_action_probs(self, state_np, valid_actions, training=None):
        state = tf.convert_to_tensor([state_np], dtype=tf.float32)
        probs = self.policy(state, training=training)
        action_probs = tfp.distributions.Categorical(probs=probs)


        return action_probs

    def choose_action(self, state_np, valid_actions):
        action_probs = self.get_action_probs(state_np, valid_actions, training=False)
        action = action_probs.sample()

        return action.numpy()[0]

    def store_transition(self, state_np, action_np, reward_np):
        self.state_memory.append(state_np)
        self.action_memory.append(action_np)
        self.reward_memory.append(reward_np)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory, dtype=tf.float32)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state_np) in enumerate(zip(G, self.state_memory)):
                action_probs = self.get_action_probs(state_np, training=True)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

            grads = tape.gradient(loss, self.policy.trainable_variables)
            self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.state_memory, self.action_memory, self.reward_memory = [], [], []
