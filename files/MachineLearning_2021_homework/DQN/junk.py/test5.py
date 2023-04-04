import copy
import logging
import itertools
import sys
import os.path
import numpy as np
import pandas as pd
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim


class MemorySys:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(0, capacity), columns=['s', 'a', 'r', 's+', 't?'])
        self.i = 0  # index
        self.cap = capacity  # capacity upper bound

    def store(self, *args):
        # store new memory clips
        cap_eff = self.cap * 0.8
        self.memory.loc[self.i] = args
        self.i = self.i + 1
        if self.i >= cap_eff -1:
            self.memory.drop(self.memory.index[0:int(0.5 * cap_eff) - 1])  # drop memory
            self.i = int(0.5 * cap_eff - 1)  # set back index

    def sample(self, size):
        # sample from memory
        indices = np.random.choice(range(0, self.i), size=size)
        return self.memory.loc[indices, ["s", "a", "r", "s+", "t?"]]


    def clear(self):
        # annihilation
        self.memory = pd.DataFrame(index=range(0, self.cap), columns=['s', 'a', 'r', 's+', 't?'])
        self.i = 0


cuda = torch.device("cuda")


class Agent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.95
        self.epsilon = 1  # exploration
        self.memorysys = MemorySys(capacity=100000)
        """
        self.evaluate_net = nn.Sequential(           
            nn.Conv2d(6, 48, kernel_size=(8, 8), stride=(4, 4)),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Conv2d(48, 144, kernel_size=(4, 4), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(144, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_n)           
            ).to("cuda")
        """
        self.evaluate_net = nn.Sequential(
            nn.Conv2d(6, 96, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(96, 384, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_n)
        ).to(cuda)
        self.target_net = copy.deepcopy(self.evaluate_net).to(cuda)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.001)
        self.learnmode = "dqn"

    def reset(self, mode=None):
        self.mode = mode
        if mode == "train":
            self.trajectory = []

    def obs2stt(self, obs):
        stt = np.asarray(obs)
        stt = torch.flatten(torch.transpose(torch.as_tensor(stt, dtype=torch.float)[:, 36:196, :, :], 1, 3), 0, 1)
        stt = (stt / 255).unsqueeze(0)
        return stt

    def store(self, rwd):
        if len(self.trajectory) >= 8:
            if np.random.rand() < 0.1:
                stt_trj, _, _, act_trj, stt_new_trj, rwd_trj, tmnt_trj, _ = self.trajectory[-8:]
                self.memorysys.store(stt_trj.cpu(), act_trj, rwd_trj, stt_new_trj.cpu(), tmnt_trj)  # store into memory
            if rwd != 0:
                stt_trj, _, _, act_trj, stt_new_trj, rwd_trj, tmnt_trj, _ = self.trajectory[-8:]
                self.memorysys.store(stt_trj.cpu(), act_trj, rwd_trj, stt_new_trj.cpu(), tmnt_trj)  # store into memory
            """
            if rwd == -1:  # traceback: store transitions of several actions before losing a match
                traj_len = len(self.trajectory)
                iter_traceback_len = min(int((traj_len - 4) / 4), 3)
                for i in range(1, iter_traceback_len):
                    stt_trbc = self.trajectory[-8 - 4 * i]
                    act_trbc = self.trajectory[-5 - 4 * i]
                    stt_new_trbc = self.trajectory[-4 - 4 * i]
                    rwd_trbc = -0.1
                    tmnt_trbc = self.trajectory[-2 - 4 * i]
                    self.memorysys.store(stt_trbc.cpu(), act_trbc, rwd_trbc, stt_new_trbc.cpu(), tmnt_trbc)  # store into memory
            """

    def step(self, stt, rwd, tmnt):
        # follow the network     #.unsqueeze(0)
        result = self.target_net(stt.to(cuda)).view(-1)
        act = result.argmax()
        if(self.mode) == "train":
            if np.random.rand() < self.epsilon:
                act = np.random.randint(low=0,high=self.action_n)
            self.trajectory += [stt.cpu(), rwd, tmnt, act]
            self.store(rwd)
            if self.memorysys.i >= 5000 and np.random.rand() < 0.05:
                self.learn()
        return act

    def target_renew(self, target_net, evaluate_net, rate=0.05):
        for target_param, evaluate_param in zip(target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(rate * evaluate_param.data + (1 - rate) * target_param.data)

    def learn(self):
        if self.learnmode == "dqn":
            self.learn_dqn()
        elif self.learnmode == "double_dqn":
            self.learn_double_dqn()

    def learn_dqn(self):
        # replay
        batch_size = 32
        batch = self.memorysys.sample(batch_size)
        eva_new_t = torch.zeros(batch_size, device="cuda")
        eva_t = torch.zeros(batch_size, device="cuda")
        for i in range(0, batch_size):
            stt = batch.iat[i,0].to(cuda)
            stt_new = batch.iat[i, 3].to(cuda)
            act = torch.tensor(batch.iat[i,1], device="cuda")
            rwd = torch.tensor(batch.iat[i,2], device="cuda")
            tmnt = batch.iat[i,4]
            if int(tmnt) == 0:
                maxq = torch.max(self.evaluate_net(stt_new).view(-1))
                eva_new = rwd + self.gamma * maxq
            else:
                eva_new = rwd
            eva_new_t[i] = eva_new.detach()
            eva_t[i] = self.evaluate_net(stt).view(-1)[act].detach()
        loss_func = torch.nn.MSELoss()
        loss = loss_func(eva_new_t, eva_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if np.random.rand() < 0.01:
            self.target_net = copy.deepcopy(self.evaluate_net).to(cuda)
        #  self.target_renew(self.target_net, self.evaluate_net)
        self.epsilon = max(self.epsilon - 1e-5, 0.05)

    def learn_double_dqn(self):
        pass


env = FrameStack(gym.make('PongNoFrameskip-v4'), num_stack=2)
env.reset()
agent = Agent(env)
agent_path = 'agent_model.pt'


def play_episode(env, agent, max_steps=None, mode=None, render=False):
    obs, rwd, tmnt = env.reset(), 0., False
    agent.reset(mode)
    episode_rwd, elapsed_steps = 0., 0
    while True :
        stt = agent.obs2stt(obs)
        act = agent.step(stt, rwd, tmnt)
        if render:
            env.render()
        if int(tmnt) != 0:
            break
        obs, rwd, tmnt, _ = env.step(act)
        episode_rwd += rwd
        elapsed_steps += 1
        if max_steps and elapsed_steps >= max_steps:
            break
    return episode_rwd, elapsed_steps


if os.path.isfile(agent_path):
    agent.target_net, agent.evaluate_net = torch.load(agent_path)
else:
    print('Initializing the agent')


episode_rwds = []
for episode in range(0,10000):
    episode_rwd, elapsed_steps = play_episode(env, agent, max_steps=None, mode='train', render=True)
    print(episode_rwd, elapsed_steps, agent.epsilon, agent.memorysys.i)
    episode_rwds.append(episode_rwd)
    if episode%10==0:
        torch.save((agent.target_net, agent.evaluate_net), agent_path)
    if np.mean(episode_rwds[-5:]) > 0:
        break