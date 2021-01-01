import time
import numpy as np
from agent import Agent
from env import Env
import tensorflow as tf
from argparse import ArgumentParser
from copy import deepcopy

import subprocess

rc = subprocess.call("echo $LD_LIBRARY_PATH", shell=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    agent = Agent(lr=0.001, gamma=0.5, nb_actions=4)
    env = Env()

    score_history, game_score_history, avg_score_history, avg_game_score_history = [], [], [], []
    nb_episodes = 10000

    window = 100

    tb_logs_dir = 'logs/{}'.format(args.idx)
    tb_summary_writer = tf.summary.create_file_writer(tb_logs_dir)

    for i in range(nb_episodes):
        total_time_start = time.time()

        done = False
        score = 0
        state_np = env.reset()
        step = 0

        start_time = time.time()
        while not done:
            action_np = agent.choose_actions(np.expand_dims(state_np, axis=0)).numpy()[0]
            new_state_np, reward_np, done, info = env.step(action_np)
            agent.store_transition(state_np, action_np, reward_np)
            state_np = new_state_np
            score += reward_np
            if env.invalid_moves_cnt >= 100:
                done = True
            step += 1
            # if step == 100:
            #     done = True

        simulation_time = time.time() - start_time
        feedforward_time, backprop_time = agent.learn()

        score_history.append(score)
        game_score_history.append(env.game_score)
        avg_score = np.mean(score_history[-window:])
        avg_game_score = np.mean(game_score_history[-window:])
        avg_score_history.append(avg_score)
        avg_game_score_history.append(avg_game_score)

        print('Episode: {}; Score: {:.1f}; Avg score: {:.1f}; Max value: {}'.format(i, score, avg_score, env.max_value))

        with tb_summary_writer.as_default():
            tf.summary.scalar('score', score, step=i)
            tf.summary.scalar('avg score', avg_score, step=i)
            tf.summary.scalar('game score', env.game_score, step=i)
            tf.summary.scalar('avg game score', avg_game_score, step=i)
            tf.summary.scalar('max value', env.max_value, step=i)
            tf.summary.scalar('nb invalid moves', env.invalid_moves_cnt, step=i)
            tf.summary.scalar('nb steps', step, step=i)

        print('Episode length: {} ({} invalid); Times: simulation - {:.3f}; feedforward - {:.3f}; backprop - {:.3f}; total - {:.3f}'.format(
                step, env.invalid_moves_cnt, simulation_time, feedforward_time, backprop_time, time.time() - total_time_start))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)
