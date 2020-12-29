import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
from model import get_policy_network


class Agent:
    def __init__(self, lr=0.0005, gamma=0.99, nb_actions=4):
        self.lr = lr
        self.gamma = gamma
        self.nb_actions = nb_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.valid_actions_memory = []
        self.policy = get_policy_network(self.nb_actions)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    def get_action_probs(self, state_np, valid_actions, training=None):
        state = tf.convert_to_tensor([state_np], dtype=tf.float32)
        probs = self.policy(state, training=training)
        # if len(valid_actions) < 4:
        #     valid_actions_oh = tf.convert_to_tensor(to_categorical(valid_actions, num_classes=4), dtype=tf.float32)
        #     if training:
        #         valid_actions_oh = 1 - tf.expand_dims(tf.reduce_sum(valid_actions_oh, axis=0), axis=0)
        #     else:
        #         valid_actions_oh = tf.expand_dims(tf.reduce_sum(valid_actions_oh, axis=0), axis=0)
        #     probs = (probs * valid_actions_oh + 0.1) / tf.reduce_sum(probs * valid_actions_oh + 0.1)
        action_probs = tfp.distributions.Categorical(probs=probs)


        return action_probs

    def choose_action(self, state_np, valid_actions):
        action_probs = self.get_action_probs(state_np, valid_actions, training=False)
        action = action_probs.sample()

        return action.numpy()[0]

    def store_transition(self, state_np, action_np, reward_np, valid_actions):
        self.state_memory.append(state_np)
        self.action_memory.append(action_np)
        self.reward_memory.append(reward_np)
        self.valid_actions_memory.append(valid_actions)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory, dtype=tf.float32)

        Gs = []
        minus_J_delta_componentes = []

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            minus_J_delta = 0
            for idx, (g, state_np) in enumerate(zip(G, self.state_memory)):
                action_probs = self.get_action_probs(state_np, self.valid_actions_memory[idx], training=True)
                # action_probs = self.get_action_probs(state_np, tf.ones((4,)), training=True)
                log_prob = action_probs.log_prob(actions[idx])
                component = - g * tf.squeeze(log_prob)
                minus_J_delta += component

                Gs.append(g)
                minus_J_delta_componentes.append(component.numpy())

            if tf.math.is_nan(minus_J_delta):
                print('minus_J_delta is nan')
            else:
                # loss /= len(self.reward_memory)
                grads = tape.gradient(minus_J_delta, self.policy.trainable_variables)
                self.policy.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.state_memory, self.action_memory, self.reward_memory = [], [], []
        self.valid_actions_memory = []

        return minus_J_delta.numpy(), Gs, minus_J_delta_componentes
