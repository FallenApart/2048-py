import numpy as np


class Env:
    def __init__(self, nb_rows=4, nb_cols=4):
        self.nb_actions = 4
        self.nb_rows, self.nb_cols = nb_rows, nb_cols
        self.reset()

    def reset(self):
        self.max_value = 0
        self.invalid_moves_cnt = 0
        self.done = False
        self.game_score = 0
        self.valid_actions = [0, 1, 2, 3]  # 0 - right, 1 - left, 2 - up, 3 - down
        self.state = np.zeros((self.nb_rows, self.nb_cols))
        self.update_empty_cells()
        self.add_number()
        self.add_number()
        return self.state

    def step(self, action):
        if action in self.valid_actions:
            if action == 0:
                self.state, reward = self.move_right(self.state)
            elif action == 1:
                self.state, reward = self.move_left(self.state)
            elif action == 2:
                self.state, reward = self.move_up(self.state)
            elif action == 3:
                self.state, reward = self.move_down(self.state)
            self.update_empty_cells()
            self.add_number()
            self.is_done()
            self.game_score += reward
        else:
            reward = 0
            self.invalid_moves_cnt += 1
        self.max_value = np.max(self.state)
        return self.state, reward, self.done, None

    def update_empty_cells(self):
        self.empty_cells = self.state == 0

    def add_number(self):
        try:
            empty_cells_idxs = np.transpose(self.empty_cells.nonzero())
        except:
            print("An exception occurred")
        selected_cell = empty_cells_idxs[np.random.choice(empty_cells_idxs.shape[0], 1, replace=False)][0]
        r = np.random.rand(1)[0]
        self.state[selected_cell[0], selected_cell[1]] = 2 if r < 0.9 else 4
        self.update_empty_cells()
        self.update_valid_actions()

    @staticmethod
    def slide_row_right(row):
        non_empty = row[row!=0]
        missing = row.shape[0] - non_empty.shape[0]
        new_row = np.append(np.zeros(missing), non_empty)
        return new_row

    def move_row_right(self, row):
        row = self.slide_row_right(row)
        col_reward = 0
        for col in reversed(range(1, self.nb_cols)):
            if row[col] == row[col-1]:
                col_reward += row[col]
                row[col] = 2 * row[col]
                row[col-1] = 0
        row = self.slide_row_right(row)
        return row, col_reward

    def move_right(self, state):
        new_state = np.copy(state)
        reward = 0
        for row in range(self.nb_rows):
            new_state[row], col_reward = self.move_row_right(state[row])
            reward += col_reward
        return new_state, reward

    def move_left(self, state):
        new_state = np.flip(state, axis=1)
        new_state, reward = self.move_right(new_state)
        new_state = np.flip(new_state, axis=1)
        return new_state, reward

    def move_down(self, state):
        new_state = state.T
        new_state, reward = self.move_right(new_state)
        new_state = new_state.T
        return new_state, reward

    def move_up(self, state):
        new_state = state.T
        new_state = np.flip(new_state, axis=1)
        new_state, reward = self.move_right(new_state)
        new_state = np.flip(new_state, axis=1)
        new_state = new_state.T
        return new_state, reward

    def is_done(self):
        if np.sum(self.empty_cells) == 0:
            for row in range(self.nb_rows):
                for col in range(self.nb_cols):
                    if col != self.nb_cols-1 and self.state[row][col] == self.state[row][col+1]:
                        return None
                    if row != self.nb_rows-1 and self.state[row][col] == self.state[row+1][col]:
                        return None
            self.done = True

    def update_valid_actions(self):
        self.valid_actions = [0, 1, 2, 3]
        new_state, _ = self.move_right(self.state)
        if np.array_equal(new_state, self.state):
            self.valid_actions.remove(0)
        new_state, _ = self.move_left(self.state)
        if np.array_equal(new_state, self.state):
            self.valid_actions.remove(1)
        new_state, _ = self.move_up(self.state)
        if np.array_equal(new_state, self.state):
            self.valid_actions.remove(2)
        new_state, _ = self.move_down(self.state)
        if np.array_equal(new_state, self.state):
            self.valid_actions.remove(3)


if __name__ == '__main__':
    env = Env()

    scores = []

    for _ in range(10):

        step = 0
        score = 0
        env.reset()
        done = False
        while not done:
            action = np.random.randint(0, 4, size=1)[0]
            new_state, reward, done, info = env.step(action)
            step += 1
            # print('Step: {}'.format(step))
            # print(new_state)
            # score += reward
            if step == 100:
                done = True

        print('Game done, Score: {}. Max tile: {}'.format(env.game_score, env.max_value))

        scores.append(env.game_score)

    print("Avg score: {}, std: {}".format(np.mean(scores), np.std(scores)))

