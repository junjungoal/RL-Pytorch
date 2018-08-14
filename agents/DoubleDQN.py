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

class DoubleDQN(object):
    def __init__(self, env, q_net, loss_func, opt, lr=0.00025, imsize=(84, 84), gamma=0.99, tau=0.001, buffer_size=1e6, log_dir=None, weight_dir=None):
        self.env = env
        self.q_net = q_net.type(dtype)
        self.target_q_net = copy.deepcopy(q_net).type(dtype)
        self.loss_func = loss_func
        self.opt = opt(self.q_net.parameters(),  lr)
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size

        self.n_action_space = env.action_space.n
        self._state_size = env.observation_space.shape
        self._imsize = imsize

        self.train_reward_list = []
        self.test_reward_list = []
        self.train_error_list = []
        self._buffer = ReplayBuffer([1, ], self._state_size, imsize, buffer_size)

        self.log_dir = log_dir if log_dir is not None else "./logs/"
        self.weight_dir = weight_dir if weight_path is not None else "./checkpoints/"

    def update_params(self):
        self.target_q_net = copy.deepcopy(self.q_net)

    def action(self, state):
        img = Image.fromarray(state).convert('L').crop((0, 20, 160, 210))
        x = np.array(img.resize(self._imsize))
        x = torch.from_numpy(np.expand_dims(x[None, ...].astype(np.float), axis=3)).permute((0, 3, 1, 2)).type(dtype)
        action = self.q_net(x/255.).max(1)[1].item()
        return action

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
            bar = tqdm()

            for j in range(episode_step):
                if np.random.rand() < eps_greedy:
                    img = Image.fromarray(state).convert('L')
                    x = np.array(img.resize(self._imsize))
                    x = torch.from_numpy(np.expand_dims(x[None, ...].astype(np.float), axis=3).transpose((0, 3, 1, 2))).type(dtype)
                    action = self.q_net(x/255.).max(1)[1].item()
                    # take an action maximizing Q function
                else:
                    action = self.env.action_space.sample()


                eps_greedy += g_step
                eps_greedy = np.clip(eps_greedy, min_greedy, max_greedy)

                next_state, reward, terminal, _ = self.env.step(action)

                total_reward += reward
                train_each_episode_reward.append(reward)
                self._buffer.store(state, np.array(action),
                       np.array(reward), next_state, np.array(terminal))
                train_one_episode_reward.append(reward)
                state = next_state

                if len(self._buffer) > batch_size and j % train_frequency == 0:
                    train_prestate, train_action, train_reward, train_state, train_terminal = self._buffer.get_minibatch(batch_size)

                    ## target
                    x = torch.from_numpy(np.expand_dims(train_state, axis=3).transpose((0, 3, 1, 2))/ 255.).type(dtype)
                    prior_action = self.q_net(x).max(1)[1].detach()
                    next_q_value = self.target_q_net(x)
                    next_q_value = next_q_value.gather(1, prior_action.view(-1, 1).type(dlongtype)).view(-1)
                    non_terminal = torch.from_numpy((~train_terminal).astype(np.float)).type(dtype)
                    target = torch.from_numpy(train_reward).type(dtype) + next_q_value  * self.gamma * non_terminal
                    target = target.detach()
                    # -----

                    train_prestate = torch.from_numpy(np.expand_dims(train_prestate, axis=3).transpose((0, 3, 1, 2)) / 255.).type(dtype)
                    z = self.q_net(Variable(train_prestate, requires_grad=True).type(dtype))
                    z = z.gather(1, torch.Tensor(train_action).type(dlongtype))
                    l = self.loss_func(z, Variable(torch.tensor(target.reshape((-1, 1)))).type(dtype))
                    self.opt.zero_grad()
                    l.backward()
                    for param in self.q_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.opt.step()

                    loss += l.cpu().detach().numpy()

                    msg = "episode {:03d} each step reward:{:5.3f}".format(e, total_reward)
                    bar.set_description(msg)
                    bar.update(1)

                if steps % update_period == 0:
                    self.update_params()

                info = {
                    'reward_per_1000_steps': total_reward,
                }
                if steps % LOG_EVERY_N_STEPS == 0:
                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, steps+1)
                if terminal:
                    break
                steps += 1

            state = self.env.reset()
            self.train_reward_list.append(total_reward)
            self.train_error_list.append(float(loss) / (j + 1))
            best_reward = np.max(self.train_reward_list)
            if len(self.train_reward_list) > 100:
                mean_reward = np.mean(self.train_reward_list[-100:])
                if (best_reward != -float('inf')):
                    info = {
                        'mean_episode_reward_last_100': mean_reward,
                        'best_mean_episode_reward': best_reward
                    }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, e+1)
            else:
                mean_reward = np.mean(self.train_reward_list)

            msg = ("episode {:03d} avg_loss:{:6.3f} total_reward [train:{:5.3f} test:-] average_reward {:5.3f} best_reward {:5.3f} e-greedy:{:5.3f}".format(
                e, float(loss) / ((j + 1)//train_frequency), total_reward, mean_reward, best_reward, eps_greedy))

            if e % 1000 == 0:
                torch.save(self.q_net, os.path.join(self.weight_dir, "model_{:d}.h5".format(e)))

            bar.set_description(msg)
            bar.update(0)
            bar.refresh()
            bar.close()


            sleep(0.05)

