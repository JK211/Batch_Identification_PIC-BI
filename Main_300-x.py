#!/usr/bin/env python
# coding=utf-8
'''
@Author: JK211
@Email: jerryren2884@gmail.com
@Date: 2023-09-19
@LastEditor: JK211
LastEditTime: 2024-04-15
@Discription: This .py generates intelligences and trains and tests them in the Batch_Sighnatures_4actions.py environment,
                where training and testing plots can be drawn to judge the effectiveness of the proposed scheme.
@Environment: python 3.7
'''
import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # Absolute path to the current file
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # Adding a path to the system path

import torch
import numpy as np
import argparse
from common.utils import save_results_1, make_dir
from common.utils import plot_rewards, save_args
from DQN import DQN
from Batch_Signatures_4actions import BatchEnvironment

# Save the original sys.stdout
original_stdout = sys.stdout

# Specify the output file path
output_file_path = "output.txt"

np.set_printoptions(suppress=True)

veri_nums = []


def get_args():
    """ Hyperparameters
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DQN', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=3500, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--epsilon_start', default=0.99, type=float, help="initial value of epsilon")
    parser.add_argument('--epsilon_end', default=0.005, type=float, help="final value of epsilon")
    parser.add_argument('--epsilon_decay', default=2500, type=int, help="decay rate of epsilon")
    parser.add_argument('--lr', default=0.00018, type=float, help="learning rate")  # 0.00018
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--target_update', default=4, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)  # 256 300-x
    parser.add_argument('--batchSig_num', default=300, type=int)
    parser.add_argument('--invalid_Sig_num', default=40, type=int)
    parser.add_argument('--result_path', default=curr_path + "/outputs/results/")
    parser.add_argument('--model_path', default=curr_path + "/outputs/models/")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
    args.device = torch.device("cpu")  # check GPU

    return args


def env_agent_config(cfg, seed=1):
    """
    Creating environments and agent
    """
    env = BatchEnvironment(argv=cfg)  # Creating the Environment
    n_states = 2  # state dimension
    n_actions = env.action_space.n  # Dimension of Action
    print(f"n states: {n_states}, n actions: {n_actions}")
    agent = DQN(n_states, n_actions, cfg)  # create agent
    if seed != 0:  # Setting the random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
    return env, agent


def train(cfg, env, agent):
    ''' Training
    '''
    print('Start training!')
    rewards = []  # Record rewards for all rounds
    ma_rewards = []  # Record the moving average reward for all rounds
    steps = []
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # Record a reward in a round
        ep_step = 0
        state = env.reset()  # Reset the environment and return to the initial state
        last_state = state
        last_action = 10
        while True:
            action = agent.choose_action(state, last_state, last_action)  # Select Action
            last_action = action
            last_state = state
            next_state, reward, done = env.step(action)  # Update the environment and return the transition
            agent.memory.push(state, action, reward, next_state, done)  # save transition
            state = next_state  # Updating the next state
            agent.update()  # Updating agent
            ep_reward += reward  # Cumulative rewards
            ep_step += 1
            if done:
                print("The total number of verifications at the end of the {}th time is: {}".format(i_ep + 1, env.getVeri_Sum_nums()))
                veri_nums.append(env.getVeri_Sum_nums())
                break

        if (i_ep + 1) % cfg.target_update == 0:  # Agent target network update
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            print(
                f'Episode：{i_ep + 1}/{cfg.train_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f} Epislon:{agent.epsilon(agent.frame_idx):.3f}')
            print("--------------------------------------------------------------------------")
    print('Finish training!')
    env.close()
    res_dic = {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps}
    return res_dic


def test(cfg, env, agent):
    print('Start testing!')
    ############# Since the test does not require the use of the epsilon-greedy strategy, the corresponding value is set to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    ################################################################################
    rewards = []
    ma_rewards = []
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0
        ep_step = 0
        state = env.reset()
        env.set_Nt()
        last_state = state
        last_action = 10
        while True:
            action = agent.choose_action(state, last_state, last_action)
            print("The action value of the {}th eps of the {}th test is {}".format(i_ep + 1, ep_step + 1, action))
            last_action = action
            last_state = state
            next_state, reward, done = env.step(action)
            print("The length of the set to be verified after the {}th step action of the {}th test is {}:".format(i_ep + 1, ep_step + 1, len(env.getN_t())))
            state = next_state
            ep_reward += reward
            ep_step += 1
            if done:
                print("The total number of validations at the end of the {}th time of the test is: {}".format(i_ep + 1, env.getVeri_Sum_nums()))
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f'Episode：{i_ep + 1}/{cfg.test_eps}, Reward:{ep_reward:.2f}, Step:{ep_step:.2f}')
        print("-----------------------------------------------------------------------------------")
    print('Complete the test!')
    env.close()
    return {'rewards': rewards, 'ma_rewards': ma_rewards, 'steps': steps}


if __name__ == "__main__":
    try:
        # Open the file in write mode and redirect sys.stdout to the file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            sys.stdout = file

            print("+" + '-' * 100 + "+")
            print("+" + '-' * 100 + "+")
            print("+" + '-' * 100 + "+")

            cfg = get_args()
            # training
            env, agent = env_agent_config(cfg)
            if os.path.exists(cfg.model_path + "/dqn_checkpoint.pth"):
                agent.load(cfg.model_path)
            res_dic = train(cfg, env, agent)
            make_dir(cfg.result_path, cfg.model_path)  # Create a folder to save results and model paths
            save_args(cfg)
            agent.save(path=cfg.model_path)  # Save Models
            list.sort(veri_nums)
            print("The minimum number of validations explored in the training phase is:", veri_nums[0])
            print(" List of training verification times:", veri_nums)
            save_results_1(res_dic, tag='train', path=cfg.result_path)  # save results
            plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")  # plot results
            # testing
            env, agent = env_agent_config(cfg)
            agent.load(path=cfg.model_path)  # Import model
            print("The model path is:", cfg.model_path)
            res_dic = test(cfg, env, agent)
            save_results_1(res_dic, tag='test', path=cfg.result_path)
            plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="test")

    finally:
        sys.stdout = original_stdout
