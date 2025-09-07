import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from jinja2.compiler import F
from matplotlib.patheffects import Normal

env = gymnasium.make('Pendulum-v1')
a_max = env.action_space.high[0]

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        mean = a_max * torch.tanh(mean) #限制在[-2,2]
        std = self.std(x)
        std = F.softplus(std)
        #限制大小后的均值，标准差——网络训练预测数据
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std) #分布
        action = dist.sample()
