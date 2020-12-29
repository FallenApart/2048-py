import numpy as np
from agent import Agent
from env import Env
import matplotlib.pyplot as plt
import tensorflow as tf
from argparse import ArgumentParser
from copy import deepcopy
from utils import fig_to_ftimage

import subprocess

rc = subprocess.call("echo $LD_LIBRARY_PATH", shell=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    agent = Agent(lr=0.001, gamma=1, nb_actions=4)
    env = Env()

    score_history, game_score_history, avg_score_history, avg_game_score_history = [], [], [], []
    nb_episodes = 10000

    window = 100

    tb_logs_dir = 'logs/{}'.format(args.idx)
    tb_summary_writer = tf.summary.create_file_writer(tb_logs_dir)

    for i in range(nb_episodes):
        done = False
        score = 0
        state_np = env.reset()
        step = 0

        while not done:
            valid_actions = deepcopy(env.valid_actions)
            action_np = agent.choose_action(state_np, valid_actions)
            new_state_np, reward_np, done, info = env.step(action_np)
            agent.store_transition(state_np, action_np, reward_np, valid_actions)
            state_np = new_state_np
            score += reward_np
            # if env.invalid_moves_cnt >= 100:
            #     done = True
            step += 1
            if step == 100:
                done = True

        # score += 1000 * env.invalid_moves_cnt

        minus_J_delta, Gs, minus_J_delta_componentes = agent.learn()

        score_history.append(score)
        game_score_history.append(env.game_score)
        avg_score = np.mean(score_history[-window:])
        avg_game_score = np.mean(game_score_history[-window:])
        avg_score_history.append(avg_score)
        avg_game_score_history.append(avg_game_score)

        print('Episode: {}; -J_delta: {:.3f}; Score: {:.1f}; Avg score: {:.1f}'.format(i, minus_J_delta, score, avg_score))
        print('Nb of steps {}'.format(step))
        print('Max value {}'.format(env.max_value))
        print('Nb invalid moves: {}'. format(env.invalid_moves_cnt))

        fig = plt.figure(figsize=(10, 10))
        plt.plot(Gs)
        Gs_img = fig_to_ftimage(fig)

        # plt.ylim(-10, 10)
        plt.plot(minus_J_delta_componentes)
        minus_J_delta_componentes_img = fig_to_ftimage(fig)

        with tb_summary_writer.as_default():
            tf.summary.scalar('score', score, step=i)
            tf.summary.scalar('avg score', avg_score, step=i)
            tf.summary.scalar('game score', env.game_score, step=i)
            tf.summary.scalar('avg game score', avg_game_score, step=i)
            tf.summary.scalar('-J_delta', minus_J_delta, step=i)
            tf.summary.scalar('max value', env.max_value, step=i)
            tf.summary.scalar('nb invalid moves', env.invalid_moves_cnt, step=i)
            tf.summary.scalar('nb steps', step, step=i)

            tf.summary.image('Gs', Gs_img, step=i)
            tf.summary.image('-J_delta_components', minus_J_delta_componentes_img, step=i)

    filename = 'logs/{}/fig.png'.format(args.idx)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Score')
    ax1.plot(score_history)
    ax2.plot(avg_score_history)
    ax2.set_title('Avg score ({})'.format(window))

    plt.savefig(filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)
