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


def rgb2gray(rgb):
    return np.dot(rgb, [0.2, 0.3, 0.5])


def img_press(img):
    # 160*160 -> 40*40
    shp = np.shape(img)
    row = int(shp[0]/4)
    col = int(shp[1]/4)
    img_new = np.zeros((row,col))
    for i in range(0,row):
        for j in range(0,col):
            img_new[i,j] = np.mean(img[i*4:(i+1)*4, j*4:(j+1)*4])
    return img_new


def img_preprocess(img):
    img = img[36:196,:,:]
    img = np.apply_along_axis(rgb2gray, 2, img)
    img = img_press(img)/255
    return img


class QueObs:
    # queue of observations, length = 3
    def __init__(self):
        img = np.zeros((40,40))
        self.que =[img, img, img]

    def push(self, img):
        self.que = [img, self.que[0], self.que[1]]

    def clear(self):
        img = np.zeros((40, 40))
        self.que = [img, img, img]


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
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)

    def clear(self):
        # annihilation
        self.memory = pd.DataFrame(index=range(0, self.cap), columns=['s', 'a', 'r', 's+', 't?'])
        self.i = 0


class Agent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.epsilon = 1  # exploration
        self.memorysys = MemorySys(capacity=10000)
        self.evaluate_net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=(2,2), stride=(2,2)),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(24, 192, kernel_size=(2,2), stride=(2,2)),
            nn.AvgPool2d(kernel_size=(5,5)),
            nn.Flatten(),
            nn.Linear(192, 40),
            nn.Sigmoid(),
            nn.Linear(40, self.action_n)
            )
        self.target_net = copy.deepcopy(self.evaluate_net)
        self.optimizer = optim.Adam(self.evaluate_net.parameters(), lr=0.01)
        self.learnmode = "dqn"
        self.que_obs = QueObs()

    def reset(self, mode=None):
        self.mode = mode
        if mode == "train":
            self.trajectory = []

    def img2stt(self, img, tmnt):
        img = img_preprocess(img)
        self.que_obs.push(img)
        stt_list= self.que_obs.que
        if tmnt:
            self.que_obs.clear()
        return stt_list  #

    def step(self, stt_list, rwd, tmnt):
        # follow the network
        stt = torch.as_tensor(stt_list, dtype=torch.float).unsqueeze(0)
        result = self.evaluate_net(stt).view(-1)
        act = result.argmax()
        if(self.mode) == "train":
            if np.random.rand() < self.epsilon:
                act = env.action_space.sample()
            self.trajectory += [stt_list, rwd, tmnt, act]
            if len(self.trajectory) >= 8:
                if np.random.rand() < 0.05:
                    stt_trj, _, _, act_trj, stt_new_trj, rwd_trj, tmnt_trj, _ = self.trajectory[-8:]
                    self.memorysys.store(stt_trj, act_trj, rwd_trj, stt_new_trj, tmnt_trj)  # store into memory
                if rwd != 0:
                    stt_trj, _, _, act_trj, stt_new_trj, rwd_trj, tmnt_trj, _ = self.trajectory[-8:]
                    self.memorysys.store(stt_trj, act_trj, rwd_trj, stt_new_trj, tmnt_trj)  # store into memory
                if rwd == -1:  # traceback: store transitions of several actions before losing a match
                    traj_len = len(self.trajectory)
                    iter_traceback_len = min(int((traj_len-4)/4), 4)
                    for i in range(1, iter_traceback_len):
                        stt_trbc = self.trajectory[-8-4*i]
                        act_trbc = self.trajectory[-5-4*i]
                        stt_new_trbc = self.trajectory[-4-4*i]
                        rwd_trbc = -0.1
                        tmnt_trbc = self.trajectory[-2-4*i]
                        self.memorysys.store(stt_trbc, act_trbc, rwd_trbc, stt_new_trbc, tmnt_trbc)
            if self.memorysys.i >= 128 and self.memorysys.i % 50 == 0:
                self.learn()
        return act

    def target_renew(self, target_net, evaluate_net, rate=0.02):
        for target_param, evaluate_param in zip(target_net.parameters(), evaluate_net.parameters()):
            target_param.data.copy_(rate * evaluate_param.data + (1 - rate) * target_param.data)

    def learn(self):
        if self.learnmode == "dqn":
            self.learn_dqn()
        elif self.learnmode == "double_dqn":
            self.learn_double_dqn()

    def learn_dqn(self):
        # replay
        batch_size = 64
        stts_list, acts, rwds, stts_new_list, tmnts = self.memorysys.sample(batch_size)
        eva_new_t = torch.zeros(batch_size)
        eva_t = torch.zeros(batch_size)
        for i in range(0, batch_size):
            stt = torch.as_tensor(stts_list[i], dtype=torch.float).unsqueeze(0)
            act = acts[i]
            rwd = rwds[i]
            stt_new = torch.as_tensor(stts_new_list[i], dtype=torch.float).unsqueeze(0)
            tmnt = tmnts[i]
            if int(tmnt) == 0:
                maxq = torch.max(self.target_net(stt_new).view(-1))
                eva_new = rwd + self.gamma * maxq
            else:
                eva_new = rwd
            eva_new_t[i] = eva_new
            eva_t[i] = self.evaluate_net(stt).view(-1)[act]
        loss_func = torch.nn.MSELoss()
        loss = loss_func(eva_new_t, eva_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.target_renew(self.target_net, self.evaluate_net)
        self.epsilon = max(self.epsilon - 1e-5, 0.05)

    def learn_double_dqn(self):
        pass


env = gym.make("Pong-v4")
env.reset()
agent = Agent(env)
agent_path = 'agent_model.pt'


def play_episode(env, agent, max_steps=None, mode=None, render=False):
    img, rwd, tmnt = env.reset(), 0., False
    agent.reset(mode)
    episode_rwd, elapsed_steps = 0., 0
    while True :
        stt = agent.img2stt(img, tmnt)
        act = agent.step(stt, rwd, tmnt)
        if render:
            env.render()
        if int(tmnt) != 0:
            break
        img, rwd, tmnt, _ = env.step(act)
        episode_rwd += rwd
        elapsed_steps += 1
        if max_steps and elapsed_steps >= max_steps:
            break
    return episode_rwd, elapsed_steps

episode_rwds = []
for episode in range(0,100000):
    episode_rwd, elapsed_steps = play_episode(env, agent, max_steps=None, mode='train', render=True)
    episode_rwds.append(episode_rwd)
    if episode%100==0:
        torch.save(agent, agent_path)
    print(episode_rwd, elapsed_steps)
    if np.mean(episode_rwds[-5:]) > 0:
        break








# nn

"""
from torch import exp
from torch import matmul
from torch import transpose
from torch import randn
from torch import zeros
from torch import pow
from torch import flatten

import scipy as sp

# Here all objects are torch tensors, for unity.

def sig(x):
    # sigmoid
    return (1/(1+exp(-x)))

def sigderi(x):
    # sigmoid derivatives
    return (exp(-x)/pow(1+exp(-x), 2))

def linlayer(x,w):
    # merely a linear layer
    return matmul(x,w)

def net_para_setup(num_layer, width_layer, width_input, sigma):
    # num_layer: number of layers, each layer corresponds to a linear transformation.
    # width_layer: widths of each layers, an array of length (num_layer), containing integers.
    # width_input: width of input, an integer.
    # sigma : hyper-parameter in initialization
    network_para = tuple()
    for i in range(0, num_layer):
        if i == 0:
            w = randn(size=(width_input, width_layer[0])) * sigma
            # random initialization
            network_para = network_para + tuple(w)
        else:
            w = randn(size=(width_layer[i-1], width_layer[i])) * sigma
            network_para = network_para + (w, )
    return(network_para)

def net_setup(num_layer, width_layer, width_input, sigma):
    network_para = net_para_setup(num_layer, width_layer, width_input, sigma)
    network = tuple()
    for i in range(0, num_layer):
        w = network_para[i]
        y = zeros(size=(width_layer[i]))
        grad_y = zeros(size=width_layer[i])
        if i == 0:
            x = zeros(size=width_input)
            grad_w = zeros(size=(width_input, width_layer[0]))
        else:
            x = zeros(size=width_layer[i-1])
            grad_w = zeros(size=(width_layer[i-1], width_layer[i]))
        # initialize as 0
        dic = {
            "w": w,
            "y": y,
            "gw": grad_w,
            "gy": grad_y,
            "x": x
        }
        network = network + (dic, )
    return(network)

def propagate_forward(network, input):
    # Check that the input is congruent with the network!
    num = len(network)
    network_new = tuple()
    for i in range(0, num):
        if i == 0:
            x = input
            w = (network[0])["w"]
            y = flatten(matmul(x, w))
        else:
            x = sig((network_new[i-1])["y"])
            w = (network[i])["w"]
            y = flatten(matmul(x, w))
        grad_w = zeros(size=(x.size(0), y.size(0)))
        grad_y = zeros(size=y.size(0))
        dic = {
            "w": w,
            "y": y,
            "gw": grad_w,
            "gy": grad_y,
            "x": x
        }
        network_new = network_new + (dic, )
    return(network_new)



def propagate_backward(network, input, partial, alpha):
    # partial : \frac{\partial L}{\partial y}, a dim-1 tensor
    # Check that the final partial is congruent with the network!
    # alpha: learning rate
    num = len(network)
    network_new = tuple()
    for i in range(num - 1, -1, -1):
        x = (network[i])["x"]
        y = (network[i])["y"]
        w = (network[i])["w"]
        if i == num-1:
            grad_y = partial
        else :
            grad_y = sigderi(y) * matmul((network[i+1])["w"], network[i+1]["gy"])
        grad_w = matmul(x.reshape(-1,1), grad_y)
        # Is ihe order correct for grad_w?
        dic={
            "w": w,
            "y": y,
            "gw": grad_w,
            "gy": grad_y,
            "x": x
        }
        network_new = (dic, ) + network_new
    for i in range(num - 1, -1, -1):
        (network_new[i])["w"] = (network_new[i])["w"] - alpha * (network_new[i])["gw"]
    return (network_new)
"""

"""
import gym
env = gym.make('Pong-ram-v0')
print(env.observation_space)
print(env.action_space)
"""


"""
env.reset()
env.step(env.action_space.sample())
"""