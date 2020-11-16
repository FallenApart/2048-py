import numpy as np


class Env:
    def __init__(self, nb_rows=4, nb_cols=4):
        self.nb_rows, self.nb_cols = nb_rows, nb_cols
        self.reset()

    def reset(self):
        self.done = False
        self.reward = 0
        self.state = np.zeros((self.nb_rows, self.nb_cols))
        self.update_empty_cells()
        self.add_number()
        self.add_number()
        print(self.state)

    def step(self, action):
        self.move_right()
        self.update_empty_cells()
        self.add_number()
        print(self.state)
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
        for row in range(self.nb_rows):
            self.state[row] = self.move(self.state[row])



if __name__ == '__main__':
    env = Env()

    for i in range(5):
        _, _, _, _ = env.step(None)

