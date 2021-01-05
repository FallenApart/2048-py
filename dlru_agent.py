from abc_agent import ABCAgent
import numpy as np


class DLRUAgent(ABCAgent):
    def __init__(self, nb_actions=4, env=None):
        super(DLRUAgent, self).__init__(nb_actions=nb_actions)
        self.env = env

    def choose_actions(self, states):
        #TODO Not available for batches
        new_state, _ = self.env.move_down(states[0])
        if not np.array_equal(states[0], new_state):
            return [3]
        else:
            new_state, _ = self.env.move_left(states[0])
            if not np.array_equal(states[0], new_state):
                return [1]
            else:
                new_state, _ = self.env.move_right(states[0])
                if not np.array_equal(states[0], new_state):
                    return [2]
                else:
                    new_state, _ = self.env.move_up(states[0])
                    if not np.array_equal(states[0], new_state):
                        return [0]
                    else:
                        print('Error! No valid moves')