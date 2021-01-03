from abc import ABC, abstractmethod


class ABCAgent(ABC):
    def __init__(self, nb_actions=4):
        self.nb_actions = nb_actions
        self.state_memory, self.action_memory, self.reward_memory = [], [], []

    @abstractmethod
    def choose_actions(self, states):
        pass

    def store_transition(self, state_np, action_np, reward_np):
        self.state_memory.append(state_np)
        self.action_memory.append(action_np)
        self.reward_memory.append(reward_np)
