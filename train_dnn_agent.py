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
    logs_dir = 'logs/{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dnn_name, args.gamma, args.lr, args.max_invalid_moves,
                                                     args.normalisation, args.batch_size, args.mode, args.idx)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, 'model'), exist_ok=True)

    agent = DNNAgent(mode=args.mode, lr=args.lr, gamma=args.gamma, nb_actions=4, dnn_name=args.dnn_name)
    env = Env(normalisation=args.normalisation)

    score_history, game_score_history, avg_score_history, avg_game_score_history = [], [], [], []
    nb_episodes = 100000

    window = 100

    tb_logs_dir = os.path.join(logs_dir, 'tb_logs')
    tb_summary_writer = tf.summary.create_file_writer(tb_logs_dir)

    best_score = 0

    i = 0

    while i < nb_episodes:
        batch_time_start = time.time()
        batch_state_memory, batch_action_memory, batch_reward_memory, batch_terminal_state = [], [], [], []
        batch_scores = []
        for j in range(args.batch_size):
            done = False
            state_np = env.reset()
            step = 0
            while not done:
                action_np = agent.choose_actions(np.expand_dims(state_np, axis=0))[0]
                new_state_np, reward_np, done, info = env.step(action_np)
                score_np = env.score * args.normalisation
                agent.store_transition(state_np, action_np, reward_np, score_np)
                state_np = new_state_np
                if env.invalid_moves_cnt >= args.max_invalid_moves:
                    print("Max invalid mover reached")
                    done = True
                step += 1

            terminal_state = state_np

            batch_scores.append(score_np)
            batch_state_memory.append(agent.state_memory)
            batch_action_memory.append(agent.action_memory)
            batch_reward_memory.append(agent.reward_memory)
            batch_terminal_state.append(terminal_state)
            agent.reset_memory()

            if score_np > best_score:
                best_score = env.score * args.normalisation
                # agent.policy.save(os.path.join(logs_dir, 'model', 'model.hdf5'))
                # print('New best score: {}; New model has been saved'.format(best_score))

            score_history.append(env.score * args.normalisation)
            avg_score = np.mean(score_history[-window:])
            avg_score_history.append(avg_score)

            print('Episode: {}; Score: {}; Avg score: {:.1f}; Max value: {}'.format(
                i, score_history[-1], avg_score, int(env.max_value * args.normalisation)))

            with tb_summary_writer.as_default():
                tf.summary.scalar('score', score_np, step=i)
                tf.summary.scalar('avg score', avg_score, step=i)
                tf.summary.scalar('max value', env.max_value * args.normalisation, step=i)
                tf.summary.scalar('nb invalid moves', env.invalid_moves_cnt, step=i)
                tf.summary.scalar('nb steps', step, step=i)
            i += 1

        agent.learn(batch_state_memory, batch_action_memory, batch_reward_memory, batch_terminal_state)

        print('Batch time: {:.2f} sec.; Avg batch game score: {:.1f}'.format(time.time() - batch_time_start,
                                                                             np.array(batch_scores).mean()))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dnn_name', type=str, default='dnn5')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--max_invalid_moves', type=int, default=1000)
    parser.add_argument('--normalisation', type=float, default=1024.0)
    parser.add_argument('--mode', type=str, default='a2c')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    main(args)
