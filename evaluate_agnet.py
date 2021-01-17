import numpy as np
import tensorflow as tf
from tqdm import tqdm
from argparse import ArgumentParser
from rand_agent import RandAgent
from dnn_agent import DNNAgent
from dlru_agent import DLRUAgent
from env import Env


def main(args):
    env = Env(normalisation=args.normalisation)

    if args.agent_type == 'rand':
        agent = RandAgent(nb_actions=4)
    elif args.agent_type == 'dlru':
        agent = DLRUAgent(nb_actions=4, env=Env(normalisation=args.normalisation))
    elif args.agent_type == 'dnn':
        agent = DNNAgent(nb_actions=4, dnn_name=args.dnn_name)
        agent.policy = tf.keras.models.load_model('example/model.hdf5')

    scores = []
    for idx in tqdm(range(args.nb_episodes), desc='Episodes'):
        step = 0
        state_np = env.reset()
        done = False
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
        scores.append(score_np)

        agent.terminal_state = state_np
        agent.dump_trajectory(args.normalisation, idx)

        agent.reset_memory()

    print("Avg score: {}, std: {}".format(np.mean(scores), np.std(scores)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--agent_type', type=str, default='dnn')
    parser.add_argument('--nb_episodes', type=int, default=1)
    parser.add_argument('--max_invalid_moves', type=int, default=2000)
    parser.add_argument('--dnn_name', type=str, default='dnn5')
    parser.add_argument('--normalisation', type=float, default=1024.0)
    args = parser.parse_args()
    main(args)
