#!/usr/bin/env python
# coding=utf-8
"""
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2022-07-13 00:08:18
@Discription: This .py is the DQN algorithm we copied from github
@Environment: python 3.7.7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        """ Initialize the q network as a fully connected network
            n_states: The number of input features is the state dimension of the environment
            n_actions: Output action dimensions
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  # input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, n_actions)  # output layer

    def forward(self, x):
        # The activation function corresponding to each layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  #  Capacity of Experience Replay
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' Buffer is a queue, capacity exceeded to remove the beginning of the deposit of the transfer (transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        ''' Returns the amount currently stored
        '''
        return len(self.buffer)


class DQN:
    def __init__(self, n_states, n_actions, cfg):

        self.n_actions = n_actions
        self.device = cfg.device
        self.gamma = cfg.gamma
        # Parameters related to the e-greedy policy
        self.frame_idx = 0  #  Decay counting for epsilon
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
                                         (cfg.epsilon_start - cfg.epsilon_end) * \
                                         math.exp(-1. * frame_idx / cfg.epsilon_decay)
        self.batch_size = cfg.batch_size
        self.policy_net = MLP(n_states, n_actions).to(self.device)
        self.target_net = MLP(n_states, n_actions).to(self.device)
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # Copy parameters to target network targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state, last_state, last_action):
        ''' Select Action
        '''
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                if (state == last_state).all():

                    state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()  # Select the action with the largest Q value
                    if action == last_action:
                        action = random.randrange(self.n_actions)

                else:
                    state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                    q_values = self.policy_net(state)
                    action = q_values.max(1)[1].item()

        else:
            # print("++++++++++++++++++++ this +++++++++++++++")
            action = random.randrange(self.n_actions)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:  # Do not update the policy when a batch is not satisfied in memory
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # Compute Q(s_t, a) corresponding to the current state (s_t, a)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()  # Calculate the Q value corresponding to the state (s_t_, a) at the next moment in time
        # Calculate the expected Q-value, for the terminated state, when done_batch=1, the corresponding expected_q_value is equal to forward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # Calculation of root mean square loss
        # print("---------loss值----------", loss)
        # Optimize and update models
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip prevents gradient explosion
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
