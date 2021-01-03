import numpy as np
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
from rand_agent import RandAgent
from dnn_agent import DNNAgent
from env import Env


def main(args):
    env = Env()

    if args.agent_type == 'rand':
        agent = RandAgent(nb_actions=4)
    elif args.agent_type == 'dnn':
        agent = DNNAgent(nb_actions=4, dnn_name=args.dnn_name)
        agent.policy = tf.keras.models.load_model('model.hdf5')

    scores = []
    for _ in tqdm(range(args.nb_episodes), desc='Episodes'):
        step = 0
        state_np = env.reset()
        done = False
        while not done:
            action = agent.choose_actions(np.expand_dims(state_np, axis=0))
            new_state, reward, done, info = env.step(action)
            if env.invalid_moves_cnt >= args.max_invalid_moves:
                done = True
            step += 1
        scores.append(env.game_score * 1024)

    print("Avg score: {}, std: {}".format(np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='dnn')
    parser.add_argument('--nb_episodes', type=int, default=10)
    parser.add_argument('--max_invalid_moves', type=int, default=2000)
    parser.add_argument('--dnn_name', type=str, default='dnn5')
    args = parser.parse_args()
    main(args)
