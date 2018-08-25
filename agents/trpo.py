# coding: utf-8
import sys, os
sys.path.append("..")
import numpy as np
import copy

from matplotlib import pyplot as plt
from tqdm import tqdm
import gym
from utils.logger import Logger
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.replay_buffer import ReplayBuffer
from PIL import Image
from time import sleep

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class TRPO(object):
    def __init__(self, env, policy_net, value_net,
                 loss_func, opt, lr=0.00025, imsize=(84, 84),
                 gamma=0.99, tau=0.001, buffer_size=1e5,
                 log_dir=None, weight_dir=None):
        self.env = env

        self.policy_net = policy_net.type(dtype)
        self.value_net = value_net.type(dtype)

        self.loss_func = loss_func
        self.opt = opt(self.policy_net.parameters(),  lr)
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size

        self._state_size = env.observation_space.shape
        self._imsize = imsize

        self.train_reward_list = []
        self.test_reward_list = []
        self.train_error_list = []
        self._buffer = ReplayBuffer([1, ], self._state_size, imsize, buffer_size)

        self.log_dir = log_dir if log_dir is not None else "./logs/"
        self.weight_dir = weight_dir if weight_dir is not None else "./checkpoints/"

    def select_action(self, state):
        state = torch.from_numpy(state).type(dtype).unsqueeze(0)
        prob = self.policy_net(Variable(state))
        action = prob.multinomial(1)
        return action, prob

    def sample_trajectories(self):
        raise NotImplementedError

    def grayscale(self, state):
        state = Image.fromarray(state).convert('L')
        state = np.array(state.resize(self._imsize))
        return state

    def train(self,
              episode=50000,
              batch_size=32,
              episode_step=10000,
              random_step=50000,
              min_greedy=0.0,
              max_greedy=0.9,
              greedy_step=1000000,
              test_step=1000,
              update_period=10000,
              train_frequency=4,
              test_eps_greedy=0.05,
              test_period=10):

        LOG_EVERY_N_STEPS = 1000
        logger = Logger(self.log_dir)

        state = self.env.reset()
        steps = 0

        eps_greedy = min_greedy
        g_step = (max_greedy - min_greedy) / greedy_step

        running_state = ZFilter(self._imsize, clip=5)
        running_reward = ZFilter((1,), demean=False, clip=10)

        for e in range(episode):
            loss = 0

            train_one_episode_reward = []
            train_each_episode_reward = []
            test_one_episode_reward = []
            test_each_episode_reward = []

            reward_batch = 0
            num_steps = 0
            num_episodes = 0

            entropy = 0
            while len(self._buffer) < batch_size:
                state = self.env.reset()
                state = self.grayscale(state)
                state = running_state(state)

                total_reward = 0
                for i in range(episode_step):
                    state = np.expand_dims(state, axis=0)
                    action, action_dist = self.select_action(state)
                    actions.append(action)
                    action_distributions.append(action_dist)

                    entropy += -(action_dist * action_dist.log()).sum()

                    next_state, reward, terminal, _ = self.env.step(action)
                    total_reward += reward
                    next_state = self.grayscale(next_state)
                    self._buffer.store(state, np.array(action),
                                       np.array(reward), next_state, np.array(terminal))
                    next_state = running_state(next_state)
                    state = next_state

                    if terminal:
                        state = self.env.reset()

                rewrd_batch += total_reward
                num_episodes += 1
                num_steps += (i-1)

            import pdb
            pdb.set_trace()
            reward_batch /= num_episodes
            train_prestate, train_action, train_reward, train_state, train_terminal = self._buffer.get_minibatch(batch_size)
            train_prestate = torch.Tensor(train_prestate)
            train_action = torch.Tensor(train_action)
            train_reward = torch.Tensor(train_reward)
            train_state = torch.Tensor(train_state)
            train_terminal = torch.Tensor(train_terminal)

            returns = torch.Tensor(train_actions.size(0),1)
            deltas = torch.Tensor(train_actions.size(0),1)
            advantages = torch.Tensor(train_actions.size(0),1)



    def test(self):
        raise NotImplementedError

class ZFilter(object):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x


# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

