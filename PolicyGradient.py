import gymnasium
import torch
#神经网络类式接口
import torch.nn as nn
#神经网络函数接口
import torch.nn.functional as F

#状态->动作概率
class Policy(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=36, action_dim=2): #隐藏层神经元数量暂定36 4*9
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x): #x必须是张量
        x = F.relu(self.fc1(x)) #输入映射ReUL
        output = F.softmax(self.fc2(x), dim=-1) #dim=-1直接选中特征数，无需记忆索引
        return output #动作概率分布

env = gymnasium.make("CartPole-v1") #预设环境
policy = Policy()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

gamma = 0.99 #折扣因子γ
for episode in range(500): #episode一局
    state = env.reset() #初始化环境[位置，速度，角度，角速度]
    log_prob = [] #logΠ(a|s)
    rewards = []

    done = False
    while True:
        #算法
        state_tensor = torch.tensor(state, dtype=torch.float) #转化为张量，PyTorch定义的神经网络只接收张量输入
        action_prob = policy.forward(state_tensor) #映射得概率

        
        ####
        next_observation, reward, terminated, truncated, info = env.step(action) #执行动作，映射环境定义
        done = terminated or truncated
        if done:
            break

        update(observation, action, reward, terminated) #算法更新模型参数
        observation = next_observation #更新观测空间

env.close()

