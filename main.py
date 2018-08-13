import sys, os
import click
from agents.DQN import DQN
from agents.DoubleDQN import DoubleDQN
from models.QNet import QNet

import gym
import torch
import torch.nn as nn
import torch.optim as optim


@click.group()
def main():
    pass

@main.command()
@click.option('-e', '--env', help='environment to be trained', required=True)
@click.option('-l', '--learning_rate', help='Learning rate', default=0.00025)
@click.option('-b', '--batch_size', help='Batch size', default=32)
@click.option('-r', '--random_step', help='Random Steps', default=50000)
@click.option('--log_dir', help='log directory', default=None)
def dqn(env, learning_rate, batch_size, random_step, log_dir):
    print('Env Name: ', env)
    env = gym.make(env)
    print('Action Space: ', env.action_space.n)
    print('State Shape:', env.render(mode='rgb_array').shape)
    agent = DQN(env,
                QNet(env.action_space.n),
                nn.MSELoss(),
                optim.RMSprop,
                lr=learning_rate,
                log_dir=log_dir)
    agent.train(batch_size=batch_size, random_step=random_step)

@main.command()
@click.option('-e', '--env', help='environment to be trained', required=True)
@click.option('-l', '--learning_rate', help='Learning rate', default=0.00025)
@click.option('-b', '--batch_size', help='Batch size', default=32)
@click.option('-r', '--random_step', help='Random Steps', default=50000)
@click.option('--log_dir', help='log directory', default=None)
def double_dqn(env, learning_rate, batch_size, random_step, log_dir):
    print('Env Name: ', env)
    env = gym.make(env)
    print('Action Space: ', env.action_space.n)
    print('State Shape:', env.render(mode='rgb_array').shape)
    agent = DoubleDQN(env,
                QNet(env.action_space.n),
                nn.MSELoss(),
                optim.RMSprop,
                lr=learning_rate,
                log_dir=log_dir)
    agent.train(batch_size=batch_size, random_step=random_step)


if __name__ == '__main__':
    main()
