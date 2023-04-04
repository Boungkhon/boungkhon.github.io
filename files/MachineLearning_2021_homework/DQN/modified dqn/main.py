import copy
from collections import namedtuple
from collections import deque
from itertools import count
import math
import random
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym

import cv2


# copied from "wrappers.py"

class RewardScaler(gym.RewardWrapper):

    def reward(self, reward):
        return reward * 0.1


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs


def make_env(env, stack_frames=True, episodic_life=True, clip_rewards=False, scale=False):
    if episodic_life:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env)
    if stack_frames:
        env = FrameStack(env, 4)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env



# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float() / 255
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.head(x)
        return x


Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))

def step(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return evaluat_net(state.to(device)).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

"""
def optimize():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)
    state_action_values = evaluat_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in evaluat_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
"""

def obs2state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

"""
def train(env, n_episodes, render=True):
    for episode in range(n_episodes):
        obs = env.reset()
        state = obs2state(obs)
        total_reward = 0.0
        for t in count():
            action = step(state)
            if render:
                env.render()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = obs2state(obs)
            else:
                next_state = None
            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))  # store memories
            state = next_state
            if steps_done > INITIAL_MEMORY:
                optimize()
                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(evaluat_net.state_dict())
            if done:
                break
        if episode % 20 == 0:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return
"""

def test(env, n_episodes, policy, render=True):
    # env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
    for episode in range(n_episodes):
        obs = env.reset()
        state = obs2state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to(device)).max(1)[1].view(1, 1)
            if render:
                env.render()
                time.sleep(0.02)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                next_state = obs2state(obs)
            else:
                next_state = None
            state = next_state
            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break
    env.close()
    return




# hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
TARGET_UPDATE = 1000
RENDER = True
lr = 1e-4
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY

# initial setting
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000

"""
# for pre-trained model
EPS_START = 0.02
EPS_END = 0.02
EPS_DECAY = 1000000
"""

"""
# create networks
evaluat_net = DQN(n_actions=4).to(device)
target_net = DQN(n_actions=4).to(device)
target_net.load_state_dict(evaluat_net.state_dict())
"""

# load trained networks
# evaluat_net = torch.load("dqn_pong_model")
# target_net = torch.load("dqn_pong_model")

evaluat_net = DQN(n_actions=4).to(device)
evaluat_net.load_state_dict(torch.load("dqn_pong_model_dict"))



# optimizer
optimizer = optim.Adam(evaluat_net.parameters(), lr=lr)

steps_done = 0

# environment
env = gym.make("PongNoFrameskip-v4")
env = make_env(env)

# memory
memory = Memory(MEMORY_SIZE)

"""
# train model
train(env, 200)
torch.save(evaluat_net, "dqn_pong_model")
"""

# test model
# evaluat_net = torch.load("dqn_pong_model")
test(env, 1, evaluat_net, render=True)


