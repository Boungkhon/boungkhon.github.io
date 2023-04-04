import copy
import itertools
import logging
import os
import random
import sys

import pandas as pd
import torch
import gym
import numpy as np
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import matplotlib.pyplot as plt

from torch import nn
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')


class conv_net(nn.Module):
    def __init__(self):
        super(conv_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(5 * 3 * 64, 512)
        self.linear2 = nn.Linear(512, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # output=self.softmax(x)
        return x


class DQN_working_memory:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity), columns=['state', 'action', 'reward', 'next_state', 'done'])
        self.good_memory = pd.DataFrame(index=range(capacity),
                                        columns=['state', 'action', 'reward', 'next_state', 'done'])
        self.i = 0
        self.good_i = 0
        self.good_count = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def store_good(self, *args):
        self.good_memory.loc[self.good_i] = args
        self.good_i = (self.good_i + 1) % self.capacity
        self.good_count = min(self.good_count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, fields]) for fields in self.memory.columns)

    def good_sample(self, size):
        indices = np.random.choice(self.good_count, size=size)
        return (np.stack(self.good_memory.loc[indices, fields]) for fields in self.good_memory.columns)


env = gym.make('Pong-v0')
device = torch.device('cuda')


class agent:
    def __init__(self, env, device):
        self.action_n = env.action_space.n
        self.gamma = 0.5
        self.device = device
        self.epsilon = 0.9
        self.learn_time = 0
        self.memory_replay = DQN_working_memory(capacity=100000)
        self.target_net = conv_net().to(device=device)
        self.evaluate_net = conv_net().to(device=device)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.001)

    def reset(self, mode=None):
        self.mode = mode
        if mode == 'train':
            self.trajectory = []

    def set_mode(self, mode):
        self.mode = mode

    def choose_action(self, observation):
        obser_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(3, 210,
                                                                                                  160).unsqueeze(0)
        # print(obser_tensor.size())
        action_prob_tensor = self.target_net(obser_tensor).to(self.device)

        action = int(action_prob_tensor.argmax(dim=1))
        if self.mode == 'train':
            if np.random.rand() < self.epsilon:
                action = np.random.randint(low=0, high=self.action_n)
        return action

    def store_memory(self, state, action, reward, pre_state, done, mode=None):
        # print(torch.tensor(state).size())
        if mode == 'good':
            self.memory_replay.store_good(pre_state, action, reward, state, done)
        else:
            self.memory_replay.store(pre_state, action, reward, state, done)
        if self.memory_replay.i % 10 == 0 and self.memory_replay.i > 5000:
            self.learn()
            self.learn_time = (self.learn_time + 1) % 1000
            if self.learn_time == 999:
                self.target_net = copy.deepcopy(self.evaluate_net)

    def learn(self):
        batch_size = 32
        if self.memory_replay.count > 10 * batch_size:
            # if np.random.rand()<0.5:
            state, action, reward, next_state, done = self.memory_replay.sample(batch_size)
            # else:
            #    state, action, reward, next_state, done = self.memory_replay.good_sample(batch_size)
            state_tensor = torch.tensor(state, dtype=torch.float, device='cuda').view(batch_size, 3, 210, 160)
            action_tensor = torch.tensor(action, device='cuda')
            reward_tensor = torch.tensor(reward, dtype=torch.float, device='cuda')
            next_state_tensor = torch.tensor(next_state, dtype=torch.float, device='cuda').view(batch_size, 3, 210, 160)
            # print(next_state_tensor.size())
            # done_tensor=torch.tensor(done,device='cuda')
            y = torch.zeros((batch_size, 1), device='cuda')
            q_tensor = torch.zeros((batch_size, 1), device='cuda')
            for i in range(batch_size):
                action_index = int(action_tensor[i])
                # print(action_index)
                # print(self.evaluate_net(next_state_tensor[i].unsqueeze(0)).size())
                q_tensor[i][0] = self.evaluate_net(state_tensor[i].unsqueeze(0))[0][action_index]
                if done[i]:
                    y[i] = reward_tensor[i]
                else:
                    # optimal_action_index=self.evaluate_net(next_state_tensor[i].unsqueeze(0)).detach().argmax(dim=1)
                    # print(optimal_action_index)
                    y[i] = reward_tensor[i] + self.gamma * max(
                        self.evaluate_net(next_state_tensor[i].unsqueeze(0))[0]).detach()
            # print(y)
            # print(q_tensor)

            lossfun = torch.nn.MSELoss()
            loss = lossfun(y, q_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon = max(self.epsilon - 0.00001, 0.05)

    def close(self):
        pass


def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.choose_action(observation)
        pre_state = observation
        # print(action)
        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        if reward == 1:
            agent.store_memory(observation, action, reward, pre_state, done, mode='good')
            agent.store_memory(observation, action, reward, pre_state, done)
        else:
            agent.store_memory(observation, action, reward, pre_state, done)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps


agent_path = 'agent_model_Qlearning.pt'
play_agent = agent(env=env, device=device)
if os.path.isfile(agent_path):
    play_agent.target_net, play_agent.evaluate_net = torch.load(agent_path)
else:
    print('Initializing the agent')

"""
episode_rewards = []
logging.info('==== test ====')
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, play_agent, mode=None, render=True)
    episode_rewards.append(episode_reward)
    #print(play_agent.memory_replay.good_count)
    logging.debug('train episode %d: reward = %.2f, steps = %d, epsilon = %.2f, memory = %d',episode, episode_reward, elapsed_steps, play_agent.epsilon, play_agent.memory_replay.i)
"""

episode_rewards = []
logging.info('==== train ====')
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, play_agent, mode='train', render=True)
    episode_rewards.append(episode_reward)
    # print(play_agent.memory_replay.good_count)
    logging.debug('train episode %d: reward = %.2f, steps = %d, epsilon = %.2f, memory = %d', episode, episode_reward,
                  elapsed_steps, play_agent.epsilon, play_agent.memory_replay.i)
    if episode % 30 == 29:
        torch.save((play_agent.target_net, play_agent.evaluate_net), agent_path)
    if np.mean(episode_rewards[-5:]) > 16.:
        break