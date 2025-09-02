import gymnasium
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from jinja2.compiler import F


class PolicyValue(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        #Π_θ定义为高斯分布（μ：最优动作方向；δ）
        self.mean = nn.Linear(hidden_dim, action_dim) #将训练映射得到的动作特征定义为均值（动作中心）：杆子偏右->负力矩；偏左->正力矩
        #这里标记的参数是log(δ)
        self.log_std = nn.Parameter(torch.zeros(action_dim)) #长度位action_dim的零张量，标记为模型参数(nn.Parameter)，pytorch自动计算梯度

        #共享一个网络+两个头
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x): #x：环境参数
        x = torch.tanh(self.fc1(x)) #输出范围[-1,1]
        x = torch.tanh(self.fc2(x))
        return x

    def get_action(self, state):
        fratures = self.forward(state)
        

