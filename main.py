import numpy as np
from agent import Agent
from env import Env
import matplotlib.pyplot as plt
import tensorflow as tf
from argparse import ArgumentParser

import subprocess

rc = subprocess.call("echo $LD_LIBRARY_PATH", shell=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    agent = Agent(lr=0.0005, gamma=0.99, nb_actions=4)
    env = Env()

    score_history, avg_score_history = [], []
    game_score_history, avg_game_score_history = [], []
    nb_episodes = 2000

    window = 100

    tb_logs_dir = 'logs/{}'.format(args.idx)
    tb_summary_writer = tf.summary.create_file_writer(tb_logs_dir)

    for i in range(nb_episodes):
        done = False
        score = 0
        game_score = 0
        state_np = env.reset()

        while not done:
            action_np = agent.choose_action(state_np)
            new_state_np, reward_np, done, info, punishment = env.step(action_np, score)
            agent.store_transition(state_np, action_np, reward_np)
            state_np = new_state_np
            game_score += reward_np
            score += reward_np + punishment
            if score < - 100:
                done = True

        score_history.append(score)
        avg_score = np.mean(score_history[-window:])
        avg_score_history.append(avg_score)

        game_score_history.append(game_score)
        avg_game_score = np.mean(game_score_history[-window:])
        avg_game_score_history.append(avg_game_score)
        print('Episode: {}; Score: {:.1f}; Avg score: {:.1f}; Game score: {:.1f}; Avg game score: {:.1f}'.format(i, score, avg_score, game_score, avg_game_score))

        with tb_summary_writer.as_default():
            tf.summary.scalar('score', score, step=i)
            tf.summary.scalar('avg score', avg_score, step=i)
            tf.summary.scalar('game score', game_score, step=i)
            tf.summary.scalar('avg game score', avg_game_score, step=i)

        agent.learn()

    filename = 'logs/{}/fig.png'.format(args.idx)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)
    ax1.set_title('Score')
    ax1.plot(score_history)
    ax2.plot(avg_score_history)
    ax2.set_title('Avg score ({})'.format(window))

    ax1.set_title('Game score')
    ax1.plot(game_score_history)
    ax2.plot(avg_game_score_history)
    ax2.set_title('Avg game score ({})'.format(window))
    plt.savefig(filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--idx', type=int, default=1)
    args = parser.parse_args()
    main(args)
