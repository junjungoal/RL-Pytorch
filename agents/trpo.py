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
        self.policy_net = policy_net
        self.value_net = value_net

        self.loss_func = loss_func
        self.opt = opt(self.q_net.parameters(),  lr)
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
        self.weight_dir = weight_path if weight_path is not None else "./checkpoints/"

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

        for i in range(random_step):
            action = self.env.action_space.sample()
            next_state, reward, terminal, _ = self.env.step(action)
            self._buffer.store(state, np.array(action),
                               np.array(reward), next_state, np.array(terminal))
            state = next_state

            if terminal:
                state = self.env.reset()

        for e in range(episode):
            loss = 0
            total_reward = 0
            state = self.env.reset()

            train_one_episode_reward = []
            train_each_episode_reward = []
            test_one_episode_reward = []
            test_each_episode_reward = []

            for j in range(episode_step):
                train_prestate, train_action, train_reward, train_state, train_terminal = self._buffer.get_minibatch(batch_size)

    def test(self):
        raise NotImplementedError
