from abc import ABC, abstractmethod
import json


class ABCAgent(ABC):
    def __init__(self, nb_actions=4):
        self.nb_actions = nb_actions
        self.state_memory, self.action_memory, self.reward_memory, self.score_memory = [], [], [], []

    @abstractmethod
    def choose_actions(self, states):
        pass

    def store_transition(self, state_np, action_np, reward_np, score_np):
        self.state_memory.append(state_np)
        self.action_memory.append(action_np)
        self.reward_memory.append(reward_np)
        self.score_memory.append(score_np)

    def reset_memory(self):
        self.state_memory, self.action_memory, self.reward_memory, self.score_memory = [], [], [], []

    def dump_trajectory(self, normalisation, idx):
        states_list = [(state.flatten() * normalisation).astype(int).tolist() for state in self.state_memory]
        actions_list = [int(action) for action in self.action_memory]
        scores_list = [int(score) for score in self.score_memory]
        terminal_state = (self.terminal_state.flatten() * normalisation).astype(int).tolist()
        trajectory = {'states': states_list, 'actions': actions_list, 'scores': scores_list,
                      'terminal_state': terminal_state}

        with open('example/trajectory_{}.json'.format(idx), 'w') as fp:
            json.dump(trajectory, fp)
