#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2024-05-12
@LastEditor: JK211
LastEditTime: 2024-05-15
@Discription: This .py is used to calculate the one-time model inference delay of the batch authentication we use,
which corresponds to the overhead in Table 4 of the paper. It is worth noting that the
cost in Table 4 of the paper was obtained on the Raspberry Pi.
@Environment: python 3.7
'''
import numpy as np
import random as rand
import torch
import argparse
from Batch_Signatures_comparison import BatchEnvironment
from DQN import DQN
import os
import time

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
    sum = 0
    for i_ep in range(cfg.test_eps):
        ep_reward = 0
        ep_step = 0
        temp_sum = 0
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
            t_mul_1 = time.perf_counter()
            next_state, reward, done = env.step(action)  # Update the environment and return the transition
            t_mul_2 = time.perf_counter()
            temp_sum = temp_sum + t_mul_2 - t_mul_1
            print("temp_sum:", temp_sum*1000, 'ms')
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
        sum = sum + temp_sum/ep_step
    env.close()
    return env.getVeri_Sum_nums(), sum/cfg.test_eps

batch_size = 300
invalide_size_sets = list(range(1, 101))

# *******************Ours PIC-BI*********************************
# Average of 10 measurements per drawing point
ours_results = []
avg_nums = 10
sum_mul = 0
for invalid_size in invalide_size_sets:
    count = 0
    for i in range(avg_nums):
        temp_Nt = batch_request_generating(batch_size, invalid_size)
        for j in range(7):
            rand.shuffle(temp_Nt)

        cfg = get_args()
        env, agent = env_agent_config(cfg)
        agent.load(path=cfg.model_path)
        # t_mul_1 = time.perf_counter()
        count_test, sum = test(cfg, env, agent, temp_Nt, invalid_size)
        count = count + count_test
        # t_mul_2 = time.perf_counter()
        sum_mul = sum_mul + sum
    avg_count = count/avg_nums
    ours_results.append(avg_count)

print("Average time of one-time model inference operation:", (sum_mul / 1000) * 1000, 'ms')