import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Policy, self).___init__()
        self.linear1 = nn.Linear(n_inputs, 64)
        self.linear2 = nn/Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 64)
        self.linear2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        state_values = self.value_head(x)
        return state_values
