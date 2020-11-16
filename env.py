import numpy as np


class Env:
    def __init__(self, nb_rows=4, nb_cols=4):
        self.nb_rows, self.nb_cols = nb_rows, nb_cols
        self.reset()
        self.step_done = False

    def reset(self):
        self.done = False
        self.reward = 0
        self.state = np.zeros((self.nb_rows, self.nb_cols))
        self.update_empty_cells()
        self.add_number()
        self.add_number()
        print(self.state)

    def step(self, action):
        while not self.step_done:
            if action == 0:
                self.move_right()
            elif action == 1:
                self.move_left()
            elif action == 2:
                self.move_up()
            elif action == 3:
                self.move_down()

        self.update_empty_cells()
        self.add_number()
        self.step_done = False
        return self.state, self.reward, self.done, None

    def update_empty_cells(self):
        self.empty_cells = self.state == 0

    def add_number(self):
        empty_cells_idxs = np.transpose(self.empty_cells.nonzero())
        selected_cell = empty_cells_idxs[np.random.choice(empty_cells_idxs.shape[0], 1, replace=False)][0]
        r = np.random.rand(1)[0]
        self.state[selected_cell[0], selected_cell[1]] = 2 if r < 0.9 else 4
        self.update_empty_cells()

    @staticmethod
    def slide(row):
        non_empty = row[row!=0]
        missing = row.shape[0] - non_empty.shape[0]
        new_row = np.append(np.zeros(missing), non_empty)
        return new_row

    def move(self, row):
        row = self.slide(row)
        for col in reversed(range(1, self.nb_cols)):
            if row[col] == row[col-1]:
                self.reward = self.reward + row[col]
                row[col] = 2 * row[col]
                row[col-1] = 0
        row = self.slide(row)
        return row

    def move_right(self):
        new_state = np.copy(self.state)
        for row in range(self.nb_rows):
            new_state[row] = self.move(self.state[row])
        if not np.array_equal(new_state, self.state):
            self.state = new_state
            self.step_done = True

    def move_left(self):
        self.state = np.flip(self.state, axis=1)
        self.move_right()
        self.state = np.flip(self.state, axis=1)

    def move_down(self):
        self.state = self.state.T
        self.move_right()
        self.state = self.state.T

    def move_up(self):
        self.state = self.state.T
        self.state = np.flip(self.state, axis=1)
        self.move_right()
        self.state = np.flip(self.state, axis=1)
        self.state = self.state.T

    def is_done(self):
        if np.sum(self.empty_cells) == 0:
            for row in enumerate(self.nb_rows):
                for col in enumerate(self.nb_cols):
                    if col != self.nb_cols-1 and self.state[row][col] == self.state[row][col+1]:
                        pass
                    if row != self.nb_rows-1 and self.state[row][col] == self.state[row+1][col]:
                        pass
            self.done = True


if __name__ == '__main__':
    env = Env()

    step = 0
    while not env.done:
        action = np.random.randint(0, 4, size=1)[0]
        new_state, _, _, _ = env.step(action)
        step += 1
        print('Step: {}'.format(step))
        print(new_state)
