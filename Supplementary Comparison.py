#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-05-15
@Discription: This .py is an experimental comparison plot of II, MRI and our proposed PIC-BI with 300-1, 300-10, 300-20
            concurrent requests and records the number of tests required for each scheme. It is worth noting that a trained model
            is provided here for validating our experimental results. The experiment carried out in Fig. 6(a) in the
            paper is consistent with here, but the specific experimental results are slightly deviated, which is normal.
            The comparison scheme BIGM does not have open source code and cannot be reproduced, but the results of
            BIGM are optimal among II, BSI and MRI, and in the paper BIGM can be drawn directly.
@Environment: python 3.7
'''

from BSI import Binary
from MRI import Multi
import numpy as np
import random as rand
from matplotlib import pyplot as plt
import matplotlib
import torch
import argparse
import os
from Batch_Signatures_comparison1 import BatchEnvironment as BatchEnvironment1
from Batch_Signatures_comparison2 import BatchEnvironment as BatchEnvironment2
from DQN import DQN

matplotlib.rc("font", family='YouYuan')
curr_path = os.path.dirname(os.path.abspath(__file__))  #  Absolute path of the current file



def get_args():
    """ Hyperparameters
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DQN', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=3, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=1, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--epsilon_start', default=0.99, type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_end', default=0.1, type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=5500, type=int, help="decay rate of epsilon")  # The larger the value, the longer the training to maintain exploration
    parser.add_argument('--lr', default=0.00018, type=float, help="learning rate")  # 0.00018
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--target_update', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--batchSig_num', default=300, type=int)
    parser.add_argument('--invalid_Sig_num', default=15, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/results/")
    # parser.add_argument('--model_path', default=curr_path + "/outputs/models/")
    parser.add_argument('--model_path', default=curr_path + "/More trained models/300-1_A/models/")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    # args.device = torch_directml.device(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU

    return args


def env_agent_config(cfg, seed=1):
    """
    Creating environments and agent
    """
    env = BatchEnvironment1(argv=cfg)  # Creating the Environment
    n_states = 2  # state dimension
    n_actions = env.action_space.n  #  Dimension of Action
    agent = DQN(n_states, n_actions, cfg)  #  Creating agent
    if seed != 0:  #  Setting a random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent

def env_agent_config2(cfg, seed=1):
    """
    Creating environments and agent
    """
    env = BatchEnvironment2(argv=cfg)  # Creating the Environment
    n_states = 2  # state dimension
    n_actions = env.action_space.n  #  Dimension of Action
    agent = DQN(n_states, n_actions, cfg)  #  Creating agent
    if seed != 0:  #  Setting a random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent


def test(cfg, env, agent, Nt):
    ############# Since the test does not require the use of the epsilon-greedy strategy, the corresponding value is set to 0 ###############
    cfg.epsilon_start = 0.0  # Initial epsilon in e-greedy strategy
    cfg.epsilon_end = 0.0  #  Termination epsilon in e-greedy strategy
    ################################################################################
    rewards = []  # Record awards for all rounds
    ma_rewards = []  # Record the move average reward for all rounds
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0
        ep_step = 0
        state = env.reset()  # Reset the environment and return to the initial state
        env.set_Nt_from_outside(Nt)
        # env.set_invalid_Sig_num_from_outside(invalid_size)
        # env.set_Nt()
        last_state = state
        last_action = 10
        while True:
            action = agent.choose_action(state, last_state, last_action)
            last_action = action
            last_state = state
            next_state, reward, done = env.step(action)  # Update the environment and return the transition
            state = next_state  # Updating the next state
            ep_reward += reward  # cumulative reward
            ep_step += 1
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
    env.close()
    return env.getVeri_Sum_nums()

# 300-1_A
Nt_1A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 300-1_B
Nt_1B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 300-10_A
Nt_10A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 300-10_B
Nt_10B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 300-20_A
Nt_20A = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 300-20_B
Nt_20B = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]


# 300-1_A
b_1A = Binary()
b_1A.Binary_Search_Identification(Nt_1A)

mri_1A = Multi(1)
mri_1A.Multi_Round_Identification(Nt_1A, 1, 1000000)

cfg = get_args()
env, agent = env_agent_config(cfg)
agent.load(path=curr_path + "/More trained models/300-1_A/models/")
our_1A = test(cfg, env, agent, Nt_1A)

print("The total number of tests for 300-1_A required by II is:", 300)
print("The total number of tests for 300-1_A required by BSI is:", b_1A.vt-1)
print("The total number of tests for 300-1_A required by MRI is", mri_1A.vt)
print("The total number of tests for 300-1_A required by PIC-BI is", our_1A)
print("----------------------------------------------------------------------------------")

# 300-1_B
b_1B = Binary()
b_1B.Binary_Search_Identification(Nt_1B)

mri_1B = Multi(1)
mri_1B.Multi_Round_Identification(Nt_1B, 1, 1000000)

cfg = get_args()
env, agent = env_agent_config(cfg)
agent.load(path=curr_path + "/More trained models/300-1_B/models/")
our_1B = test(cfg, env, agent, Nt_1B)

print("The total number of tests for 300-1_B required by II is:", 300)
print("The total number of tests for 300-1_B required by BSI is:", b_1B.vt-1)
print("The total number of tests for 300-1_B required by MRI is", mri_1B.vt)
print("The total number of tests for 300-1_B required by PIC-BI is", our_1B)
print("----------------------------------------------------------------------------------")

# 300-10_A
b_10A = Binary()
b_10A.Binary_Search_Identification(Nt_10A)

mri_10A = Multi(1)
mri_10A.Multi_Round_Identification(Nt_10A, 10, 1000000)

cfg = get_args()
env, agent = env_agent_config(cfg)
agent.load(path=curr_path + "/More trained models/300-10_A/models/")
our_10A = test(cfg, env, agent, Nt_10A)

print("The total number of tests for 300-10_A required by II is:", 300)
print("The total number of tests for 300-10_A required by BSI is:", b_10A.vt-1)
print("The total number of tests for 300-10_A required by MRI is", mri_10A.vt)
print("The total number of tests for 300-10_A required by PIC-BI is", our_10A)
print("----------------------------------------------------------------------------------")

# 300-10_B
b_10B = Binary()
b_10B.Binary_Search_Identification(Nt_10B)

mri_10B = Multi(1)
mri_10B.Multi_Round_Identification(Nt_10B, 10, 1000000)

cfg = get_args()
env, agent = env_agent_config(cfg)
agent.load(path=curr_path + "/More trained models/300-10_B/models/")
our_10B = test(cfg, env, agent, Nt_10B)

print("The total number of tests for 300-10_B required by II is:", 300)
print("The total number of tests for 300-10_B required by BSI is:", b_10B.vt-1)
print("The total number of tests for 300-10_B required by MRI is", mri_10B.vt)
print("The total number of tests for 300-10_B required by PIC-BI is", our_10B)
print("----------------------------------------------------------------------------------")

# 300-20_A
b_20A = Binary()
b_20A.Binary_Search_Identification(Nt_20A)

mri_20A = Multi(1)
mri_20A.Multi_Round_Identification(Nt_20A, 20, 1000000)

cfg = get_args()
env, agent = env_agent_config2(cfg)
agent.load(path=curr_path + "/More trained models/300-20_A/models/")
our_20A = test(cfg, env, agent, Nt_20A)

print("The total number of tests for 300-20_A required by II is:", 300)
print("The total number of tests for 300-20_A required by BSI is:", b_20A.vt-1)
print("The total number of tests for 300-20_A required by MRI is", mri_20A.vt)
print("The total number of tests for 300-20_A required by PIC-BI is", our_20A)
print("----------------------------------------------------------------------------------")

# 300-20_B
b_20B = Binary()
b_20B.Binary_Search_Identification(Nt_20B)

mri_20B = Multi(1)
mri_20B.Multi_Round_Identification(Nt_20B, 20, 1000000)

cfg = get_args()
env, agent = env_agent_config2(cfg)
agent.load(path=curr_path + "/More trained models/300-20_B/models/")
our_20B = test(cfg, env, agent, Nt_20B)

print("The total number of tests for 300-20_B required by II is:", 300)
print("The total number of tests for 300-20_B required by BSI is:", b_20B.vt-1)
print("The total number of tests for 300-20_B required by MRI is", mri_20B.vt)
print("The total number of tests for 300-20_B required by PIC-BI is", our_20B)
print("----------------------------------------------------------------------------------")

II =[300, 300, 300]
BSI = [(b_1A.vt-1 + b_1B.vt-1)/2, (b_10A.vt-1 + b_10B.vt-1)/2, (b_20A.vt-1 + b_20B.vt-1)/2]
MRI = [20, (mri_10A.vt + mri_10B.vt)/2, (mri_20A.vt + mri_20B.vt)/2]   # 20 value come from current work, We respect and accept the data of the work at the end of the code.
OUR = [(our_1A + our_1B)/2, (our_10A + our_10B)/2, (our_20A + our_20B)/2]

# Draw weight change graph, corresponding to Fig. 4(c) on page 11 of manuscript
plt.figure(dpi=300)
plt.plot(II, color='red', marker='o', linestyle='dashed', label='II values')
plt.plot(BSI, color='yellowgreen', marker='d', linestyle='dashed', label='BSI values')
plt.plot(MRI, color='dodgerblue', marker='d', linestyle='dashed', label='MRI values')
plt.plot(OUR, color='darkviolet', marker='d', linestyle='dashed', label='Ours values')
plt.xlabel('The number of illegal requests')
plt.ylabel('The number of tests required')
plt.title("Efficiency Comparision at 300-1,300-10,300-20")
plt.xticks(range(3), [1, 10, 20])
plt.legend()
plt.show()

"""
Chen J, He K, Yuan Q, et al. Batch identification game model for invalid signatures in wireless mobile networks[J]. IEEE Transactions on Mobile Computing, 2016, 16(6): 1530-1543.
Chen J, Yuan Q, Xue G, et al. Game-theory-based batch identification of invalid signatures in wireless mobile networks[C]//2015 IEEE Conference on Computer Communications (INFOCOM). IEEE, 2015: 262-270.
Liu Z, Yuan M, Ding Y, et al. Efficient small-batch verification and identification scheme with invalid signatures in VANETs[J]. IEEE Transactions on Vehicular Technology, 2021, 70(12): 12836-12846.
"""


