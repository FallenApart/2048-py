import os
import time
import numpy as np
from dnn_agent import DNNAgent
from env import Env
import tensorflow as tf
from argparse import ArgumentParser
import subprocess

rc = subprocess.call("echo $LD_LIBRARY_PATH", shell=True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    logs_dir = 'logs/{}_{}_{}_{}_{}_{}_{}'.format(args.dnn_name, args.gamma, args.lr, args.max_invalid_moves,
                                                  args.normalisation, args.loss_type, args.idx)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'model'), exist_ok=True)

    agent = DNNAgent(lr=args.lr, gamma=args.gamma, nb_actions=4, dnn_name=args.dnn_name, loss_type=args.loss_type)
    env = Env(normalisation=args.normalisation)

    score_history, game_score_history, avg_score_history, avg_game_score_history = [], [], [], []
    nb_episodes = 100000

    window = 100

    tb_logs_dir = os.path.join(logs_dir, 'tb_logs')
    tb_summary_writer = tf.summary.create_file_writer(tb_logs_dir)

    best_score = 0

    for i in range(nb_episodes):
        total_time_start = time.time()

        done = False
        score = 0
        state_np = env.reset()
        step = 0

        start_time = time.time()
        while not done:
            action_np = agent.choose_actions(np.expand_dims(state_np, axis=0))[0]
            new_state_np, reward_np, done, info = env.step(action_np)
            agent.store_transition(state_np, action_np, reward_np)
            state_np = new_state_np
            score += reward_np
            if env.invalid_moves_cnt >= args.max_invalid_moves:
                done = True
            step += 1

        simulation_time = time.time() - start_time

        if env.game_score * args.normalisation > best_score:
            best_score = env.game_score * args.normalisation
            agent.policy.save(os.path.join(logs_dir, 'model', 'model.hdf5'))
            print('New best score: {}; New model has been saved'.format(best_score))

        feedforward_time, backprop_time = agent.learn()

        score_history.append(score)
        game_score_history.append(env.game_score * args.normalisation)
        avg_score = np.mean(score_history[-window:])
        avg_game_score = np.mean(game_score_history[-window:])
        avg_score_history.append(avg_score)
        avg_game_score_history.append(avg_game_score)

        print('Episode: {}; Score: {:.1f}; Avg score: {:.1f}; Max value: {}'.format(
            i, score, avg_score, int(env.max_value * args.normalisation)))

        with tb_summary_writer.as_default():
            tf.summary.scalar('score', score, step=i)
            tf.summary.scalar('avg score', avg_score, step=i)
            tf.summary.scalar('game score', env.game_score * args.normalisation, step=i)
            tf.summary.scalar('avg game score', avg_game_score, step=i)
            tf.summary.scalar('max value', env.max_value * args.normalisation, step=i)
            tf.summary.scalar('nb invalid moves', env.invalid_moves_cnt, step=i)
            tf.summary.scalar('nb steps', step, step=i)

        print('Episode length: {} ({} invalid); Times: simulation - {:.3f}; feedforward - {:.3f}; backprop - {:.3f}; total - {:.3f}'.format(
                step, env.invalid_moves_cnt, simulation_time, feedforward_time, backprop_time, time.time() - total_time_start))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dnn_name', type=str, default='dnn3')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_invalid_moves', type=int, default=2000)
    parser.add_argument('--normalisation', type=float, default=2048.0)
    parser.add_argument('--loss_type', type=str, default='Slogp_return')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)
