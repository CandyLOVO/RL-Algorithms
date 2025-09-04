import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

env = gymnasium.make('Pendulum-v1')
a_max = env.action_space.high

class Memory(object):
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.log_prob = []
        self.value = []
        self.done = []
        self.advantage = []
        self.returns = []

class PolicyValue(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) #两层全连接处理

        # 共享一个网络+两个头
        #actor分支
        #Π_θ定义为高斯分布（μ：最优动作方向；δ）
        self.mean = nn.Linear(hidden_dim, action_dim) #将训练映射得到的动作特征定义为均值（动作中心）：杆子偏右->负力矩；偏左->正力矩
        #这里标记的参数是log(δ)，训练中被优化
        # self.log_std = nn.Parameter(torch.zeros(action_dim)) #长度位action_dim的零张量，标记为模型参数(nn.Parameter)，pytorch自动计算梯度
        self.log_std = nn.Parameter(torch.zeros(action_dim)) #减小初始动作标准差

        #critic分支
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x): #x：环境参数
        x = torch.tanh(self.fc1(x)) #输出范围[-1,1]
        x = torch.tanh(self.fc2(x))
        return x #网络训练输出,actor、critic共同输入，特征共享

    #动作采样——旧策略
    def get_action(self, state):
        fea = self.forward(state)
        #隐藏层->输出层
        mean = self.mean(fea) #动作均值
        std = self.log_std.exp() #储存的对数标准差转化为标准差
        dist = torch.distributions.Normal(mean, std) #连续动作空间中动作的概率分布
        action = dist.sample() #动作采样
        action_tanh = torch.tanh(action) * a_max #限制action大小[-1,1]，环境输入。线性映射无法估计取到的值可以映射到多少
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  # 动作对数频率log(Π_θ)
        #tanh修正项: -sum(log(1 - tanh(u)^2 + eps))
        eps = 1e-6
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + eps).sum(-1, keepdim=True)
        #抽样动作，压缩后的抽样动作，对数动作频率（.detach()不跟踪梯度，.cpu().numpy()转换为NumPy数组）
        return action_tanh, action, log_prob

    #策略评估——同一(s,a)下新策略
    def evaluate(self, state, action):
        fea = self.forward(state)
        mean = self.mean(fea)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # tanh修正项: -sum(log(1 - tanh(u)^2 + eps))
        eps = 1e-6
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + eps).sum(-1, keepdim=True)

        entropy = dist.entropy() #熵值
        value = self.value(fea) #状态价值
        #对数动作频率，状态价值，熵值
        return log_prob, value, entropy

    def gae(self, rewards, values, done, lamb, gamma):
        T = len(rewards)
        advantage = np.zeros(T) #优势函数
        returns = np.zeros(T)
        values = np.array(values)

        ad = 0
        for t in reversed(range(T)): #T-1~0 序列反转
            delta = rewards[t] + gamma * values[t+1] * (1 - done[t]) - values[t] #加入结束标记
            ad = delta + gamma * lamb * ad * (1 - done[t])
            advantage[t] = ad
        returns = advantage + values[:-1] #截取到最后一个元素
        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) #归一化，减小方差
        advantage = torch.tensor(advantage, dtype=torch.float32)
        return advantage, returns

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
h_dim = 128
clip = 0.2
c1 = 0.9
c2 = 0.01
lamb=0.98
gamma=0.99
all_rewards = []

policy = PolicyValue(s_dim, h_dim, a_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.00005) #两个网络共享优化器

def ppo(memory, batch_size):
    state = torch.tensor(memory.state).float()
    action = torch.tensor(memory.action).float()
    log_prob_old = torch.tensor(memory.log_prob).float()
    returns = torch.tensor(memory.returns).float() #回归目标 At+Vst
    advantage = torch.tensor(memory.advantage).float()

    N = state.shape[0]

    #环境交互回合，收集新数据memory---输入--->回合内，同一批数据重复次数，更新策略
    for _ in range(5):
        #数据随机打乱
        indices = torch.randperm(N) #打乱后的索引顺序
        state = state[indices] #数据按随机索引重新排列
        action = action[indices]
        log_prob_old = log_prob_old[indices]
        returns = returns[indices]
        advantage = advantage[indices]

        for start in range(0, N, batch_size):
            end = start + batch_size
            state_batch = state[start:end]
            action_batch = action[start:end]
            log_prob_old_batch = log_prob_old[start:end]
            returns_batch = returns[start:end]
            advantage_batch = advantage[start:end]

            log_prob_new, value, entropy = policy.evaluate(state_batch, action_batch)
            ratio = torch.exp(log_prob_new - log_prob_old_batch)
            l1 = ratio * advantage_batch
            l2 = torch.clip(ratio, 1-clip, 1+clip) * advantage_batch
            l_clip = -torch.mean(torch.min(l1, l2)) #策略项
            l_value = torch.mean((value.squeeze() - returns_batch)**2) #值函数项
            l_entropy = -torch.mean(entropy) #熵项
            loss = l_clip + c1 * l_value + c2 * l_entropy

            optimizer.zero_grad() #梯度清零
            loss.backward()
            optimizer.step() #更新策略网络＆价值网络的θ

#主函数
for episode in range(2000): #回合数
    memory = Memory()
    state, info = env.reset()
    sum_rewards = 0

    for t in range(200): #单回合中最大交互次数，倒立摆done始终为False，没有终止状态
        # state_tensor = torch.tensor(state).float().unsqueeze(0) #转化为张量，增加批次维度
        state_tensor = torch.tensor(state, dtype=torch.float).view(1, -1)
        #计算旧策略数据，用于“锚点”，不需要参与梯度计算，参数固定，限制新策略更新幅度。否则，反向传播时会同时更新新策略和旧策略的参数，导致log_probs_old随训练动态变化。
        with torch.no_grad():
            action_tanh, action, log_prob = policy.get_action(state_tensor)
            fea = policy.forward(state_tensor) #隐藏层输出
            value = policy.value(fea) #隐藏层->输出层

        action_env = action_tanh.detach().cpu().numpy().reshape(-1)
        state_new, reward, terminated, truncated, info = env.step(action_env) #T+1个state_new
        done = terminated or truncated

        memory.state.append(state)
        memory.action.append(action.cpu().numpy())
        memory.log_prob.append(log_prob.detach().numpy())
        memory.reward.append(reward)
        memory.value.append(value.item())
        memory.done.append(done)

        state = state_new
        sum_rewards += reward
        if done:
            break

    state_tensor = torch.tensor(state).float().unsqueeze(0)
    with torch.no_grad():
        fea = policy.forward(state_tensor)  # 隐藏层输出
        value_final = policy.value(fea).item()  # 隐藏层->输出层
    memory.value.append(value_final)

    advantage, returns = policy.gae(memory.reward, memory.value, memory.done, lamb, gamma)
    memory.advantage = advantage
    memory.returns = returns

    ppo(memory, 128)

    # ave_reward = sum_rewards/len(memory.reward)
    all_rewards.append(sum_rewards)
    if episode % 50 == 0:
        print(f"Episode {episode}, total reward: {sum_rewards}")

plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
