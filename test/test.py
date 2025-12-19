import numpy as np
import time
import argparse
import os

from env.env_core_multi_channel import EnvCore
from run.env_test_runner import EnvRunner
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
#add_argument('--max_train_times', type=int, default=int(1e4), help='maximum training times')
parser.add_argument('--episode_num', type=int, default=int(1), help='num of episodes')
parser.add_argument('--episode_length', type=int, default=int(8e5), help='number of time slots per episode')
#parser.add_argument('--deep_copy_per_episode', type=int, default=int(5), help='num of train times to deepcopy eva DQN')


parser.add_argument('--epsilon', type=float, default=0.01, help='exploration probability')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--alpha', type=float, default=0.3, help='reward balance coefficient')
parser.add_argument('--tau', type=float, default=1e-2, help='reward balance coefficient')

parser.add_argument('--obs_his_len', type=int, default=int(5), help='the length of the kept obs for input')
parser.add_argument('--buffer_size', type=int, default=int(1e5), help='the maximum size of the experience replay')
parser.add_argument('--learning_interval', type=int, default=int(100), help='learning interval for each training')
parser.add_argument('--update_interval', type=int, default=int(200), help='update interval for the evaluate network')
parser.add_argument('--minimal_train_size', type=int, default=int(4e3), help='minimal buffer size to start training')
parser.add_argument('--batch_size', type=int, default=int(64), help='batch size for each training')


parser.add_argument('--actor_net_width', type=int, default=int(64), help='Hidden net width of actor network')
parser.add_argument('--critic_net_width', type=int, default=int(128), help='Hidden net width of critic network')

parser.add_argument('--actor_lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--critic_lr', type=float, default=5e-4, help='Learning rate')






args = parser.parse_args()

def make_env(args, arr_pro):
    return EnvCore(args, arr_pro)

def main():

    num_agent = int(4)
    num_channel = int(2)
    arr_pro = num_channel

    args.num_agent = num_agent
    args.num_channel = num_channel
    args.action_dim = 2
    args.obs_dim = int(args.action_dim + 4 + 2)

    root_path = os.path.dirname(os.getcwd())
    args.save_dir = os.path.join(root_path, "results")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # args.save_dir = os.path.join(args.save_dir, args.reward_type)
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, "N" + str(args.num_agent) + "K" + str(1))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = make_env(args, arr_pro)
    runner = EnvRunner(args, env)
    runner.run()




if __name__ == "__main__":
    main()
