#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-05-15
@Discription: This .py is an experimental comparison plot of II, MRI and our proposed PIC-BI with 300 concurrent
               requests and records the number of tests required for each scheme. It is worth noting that a trained model
               is provided here for validating our experimental results. The experiment carried out in Fig. 6(a) in the
               paper is consistent with here, but the specific experimental results are slightly deviated, which is normal.
               Especially, this model appear to converge to suboptimal or poorer solutions at less than 10. Hence, 300-1,
               300-10, and 300-20's result can be get from Supplementary Comparison.py
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
from Batch_Signatures_comparison import BatchEnvironment
from DQN import DQN

matplotlib.rc("font", family='YouYuan')
curr_path = os.path.dirname(os.path.abspath(__file__))  #  Absolute path of the current file


def batch_request_generating(batch_size, invalid_size):
    Sigs2beVefi = np.zeros(batch_size, int)
    for i in range(invalid_size):
        positive_position = rand.randrange(batch_size)

        while Sigs2beVefi[positive_position] == 1:
            positive_position = rand.randrange(batch_size)

        Sigs2beVefi[positive_position] = 1

    return Sigs2beVefi

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
    parser.add_argument('--model_path', default=curr_path + "/Trained model 300-x/")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    # args.device = torch_directml.device(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU

    return args


def env_agent_config(cfg, seed=1):
    """
    Creating environments and agent
    """
    env = BatchEnvironment(argv=cfg)  # Creating the Environment
    n_states = 2  # state dimension
    n_actions = env.action_space.n  #  Dimension of Action
    agent = DQN(n_states, n_actions, cfg)  #  Creating agent
    if seed != 0:  #  Setting a random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent


def test(cfg, env, agent, Nt, invalide_size):
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
        env.set_invalid_Sig_num_from_outside(invalid_size)
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


batch_size = 300
invalide_size_sets = list(range(1, 101))

# *******************II*********************************
ii_results = []
for invalid_size in invalide_size_sets:
    ii_results.append(batch_size)

# *******************BSI*********************************
# Average of 10 measurements per drawing point
avg_nums = 10
bsi_results = []
for invalid_size in invalide_size_sets:
    count = 0
    for i in range(avg_nums):
        temp_Nt = batch_request_generating(batch_size, invalid_size)
        b = Binary()
        b.Binary_Search_Identification(temp_Nt)
        count = count + b.vt - 1
    avg_count = count/avg_nums
    bsi_results.append(avg_count)


# *******************d-MRI*********************************
# Average of 10 measurements per drawing point
avg_nums = 10
mri_results = []
group_size = 1000000  # There's no particular significance here; it's fine to initialize a larger value
for invalid_size in invalide_size_sets:
    count = 0
    for i in range(avg_nums):
        temp_Nt = batch_request_generating(batch_size, invalid_size)
        m = Multi(invalid_size)
        m.Multi_Round_Identification(temp_Nt, invalid_size, group_size)
        count = count + m.vt
    avg_count = count/avg_nums
    mri_results.append(avg_count)


# *******************Ours PIC-BI*********************************
# Average of 10 measurements per drawing point
ours_results = []
avg_nums = 10
for invalid_size in invalide_size_sets:
    count = 0
    for i in range(avg_nums):
        temp_Nt = batch_request_generating(batch_size, invalid_size)
        for j in range(7):
            rand.shuffle(temp_Nt)

        cfg = get_args()
        env, agent = env_agent_config(cfg)
        agent.load(path=cfg.model_path)
        count = count + test(cfg, env, agent, temp_Nt, invalid_size)
    avg_count = count/avg_nums
    ours_results.append(avg_count)

# Draw efficiency comparision graphs, corresponding to Fig. 6(a) on page 12 of manuscript
plt.figure(dpi=200)
plt.plot(ii_results, color='red', linestyle='-', label='II results')
plt.plot(bsi_results, color='yellowgreen', linestyle='-', label='BSI results')
plt.plot(mri_results, color='dodgerblue',  linestyle='-', label='MRI results')
plt.plot(ours_results, color='darkviolet',  linestyle='-', label='Ours results')
plt.xlabel('The number of illegal requests')
plt.ylabel('The number of tests required')
plt.title("Efficiency comparision under concurrent requests n=300")
plt.legend()
plt.show()

