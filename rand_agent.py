from abc_agent import ABCAgent
import numpy as np


class RandAgent(ABCAgent):
    def __init__(self, nb_actions=4):
        super(RandAgent, self).__init__(nb_actions=nb_actions)

    def choose_actions(self, states):
        return np.random.randint(0, self.nb_actions, size=1)[0]
