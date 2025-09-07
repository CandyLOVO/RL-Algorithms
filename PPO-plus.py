import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

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
        std = F.softplus(std) #保证std是正数
        #限制大小后的均值，标准差——网络训练预测数据
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        std = torch.clamp(std, min=1e-6) #限制std最小值
        dist = torch.distributions.Normal(mean, std) #分布
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) #对数动作概率——log_prob_old
        #抽到得到的动作，动作对应的对数动作概率
        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.forward(state)
        std = torch.clamp(std, min=1e-6)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) #原有状态动作，在新分布下的对数动作概率——log_prob_new
        entropy = dist.entropy().sum(-1, keepdim=True)
        #新分布的对数动作概率，熵值
        return log_prob, entropy

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        #下一步价值的估计值
        return value

