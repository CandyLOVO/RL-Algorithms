import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

env = gymnasium.make('Pendulum-v1')
a_max = env.action_space.high[0]

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
        self.log_std = nn.Linear(hidden_dim, action_dim)

        #critic分支
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x): #x：环境参数
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x #网络训练输出,actor、critic共同输入，特征共享

    #动作采样——旧策略
    def get_action(self, state):
        fea = self.forward(state)
        #隐藏层->输出层
        mean = self.mean(fea) #动作均值，随状态变化
        mean = torch.tanh(mean) * a_max
        # std = self.log_std.exp() #储存的对数标准差转化为标准差，固定值
        std = F.softplus(self.log_std(fea))
        std = torch.clamp(std, min=1e-6)
        dist = torch.distributions.Normal(mean, std) #连续动作空间中动作的概率分布
        action = dist.sample() #动作采样
        # action_tanh = torch.tanh(action) * a_max #限制action大小[-1,1]，环境输入。线性映射无法估计取到的值可以映射到多少
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)  #正态分布下动作对数频率log(Π_θ)
        # #tanh修正项: -sum(log(1 - tanh(u)^2 + eps))
        # eps = 1e-6
        # log_prob -= torch.log(1 - torch.tanh(action).pow(2) + eps).sum(-1, keepdim=True)

        #抽样动作，压缩后的抽样动作，对数动作频率（.detach()不跟踪梯度，.cpu().numpy()转换为NumPy数组）
        return action, log_prob

    #策略评估——同一(s,a)下新策略
    def evaluate(self, state, action):
        fea = self.forward(state)
        mean = self.mean(fea)
        mean = torch.tanh(mean) * a_max
        # std = self.log_std.exp()
        std = F.softplus(self.log_std(fea))
        std = torch.clamp(std, min=1e-6) #限制最小值
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # #改为输入action_tanh
        # eps = 1e-6
        # u = torch.atanh(torch.clamp(action_tanh / a_max, -1 + eps, 1 - eps)) #把action_tanh反推回u = atanh(action_tanh/a_max)
        # log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        # log_prob -= torch.log(1 - torch.tanh(u).pow(2) + eps).sum(-1, keepdim=True)

        entropy = dist.entropy().sum(-1, keepdim=True) #熵值
        value = self.value(fea) #当前状态价值估计
        #对数动作频率，状态价值估计，熵值
        return log_prob, value, entropy

    def gae(self, rewards, values, done, lamb, gamma):
        #修正多回合导致的长度问题
        rewards = np.array(rewards, dtype=np.float32)
        rewards = (rewards + 8.0) / 8.0
        values = np.array(values, dtype=np.float32)
        done = np.array(done, dtype=np.float32)

        # if len(values) == len(rewards):
        #     values = np.append(values, values[-1]) #没有多存，values末尾加0，让values长度比rewards多1
        # elif len(values) > len(rewards)+1:
        #     values = values[:len(rewards)+1] #多存，切片截断values
        # values = np.array(values)

        T = len(rewards)
        assert len(values) == T + 1, f"values 长度应为 {T + 1}, 实际 {len(values)}"
        advantage = np.zeros(T, dtype=np.float32) #优势函数
        ad = 0
        for t in reversed(range(T)): #T-1~0 序列反转
            delta = rewards[t] + gamma * (1 - done[t]) * values[t+1] - values[t] #加入结束标记
            ad = delta + gamma * lamb * ad * (1 - done[t])
            advantage[t] = ad
        #values: T+1 ; advantage: T ; returns: T ; rewards: T
        returns = advantage + values[:-1] #截取到最后一个元素
        returns = torch.tensor(returns, dtype=torch.float32)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) #归一化，减小方差
        advantage = torch.tensor(advantage, dtype=torch.float32)
        #优势函数，GAE估计目标值
        return advantage, returns

#-------------------------------定义PPO函数-------------------------------#
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
h_dim = 128
lr_optim = 0.0001
policy = PolicyValue(s_dim, h_dim, a_dim)
optimizer = optim.Adam(policy.parameters(), lr_optim) #两个网络共享优化器

def ppo(memory, batch_size):
    state = torch.from_numpy(np.asarray(memory.state, dtype=np.float32))
    action = torch.from_numpy(np.asarray(memory.action, dtype=np.float32))
    log_prob_old = torch.from_numpy(np.asarray(memory.log_prob, dtype=np.float32)).reshape(-1,1)
    returns = torch.from_numpy(np.asarray(memory.returns, dtype=np.float32)).reshape(-1,1) #回归目标 At+Vst
    advantage = torch.from_numpy(np.asarray(memory.advantage, dtype=np.float32)).reshape(-1,1)

    N = state.shape[0]

    #环境交互回合，收集新数据memory---输入--->回合内，同一批数据重复次数，更新策略
    for _ in range(10):
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
            action_batch = action[start:end] #未tanh的action
            log_prob_old_batch = log_prob_old[start:end]
            log_prob_old_batch = log_prob_old_batch.detach()
            returns_batch = returns[start:end]
            returns_batch = returns_batch.detach()
            advantage_batch = advantage[start:end]
            advantage_batch = advantage_batch.detach()

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

#-------------------------------主函数-------------------------------#
clip = 0.2
c1 = 0.5
c2 = 0.01
lamb= 0.95
gamma= 0.99
batch_size = 64
all_rewards = []
step_update = 2048
global_step = 0
global_memory = Memory()

for episode in range(3000): #回合数
    # memory = Memory()
    state, info = env.reset()
    sum_rewards = 0

    for t in range(200): #单回合中最大交互次数，倒立摆done始终为False，没有终止状态
        state_tensor = torch.tensor(state).float().unsqueeze(0) #转化为张量，增加批次维度
        # state_tensor = torch.tensor(state, dtype=torch.float).view(1, -1)
        #计算旧策略数据，用于“锚点”，不需要参与梯度计算，参数固定，限制新策略更新幅度。否则，反向传播时会同时更新新策略和旧策略的参数，导致log_probs_old随训练动态变化。
        with torch.no_grad():
            action, log_prob = policy.get_action(state_tensor)
            fea = policy.forward(state_tensor) #隐藏层输出
            value = policy.value(fea) #隐藏层->输出层

        action_env = action.detach().cpu().numpy().reshape(-1)
        state_new, reward, terminated, truncated, info = env.step(action_env) #T+1个state_new
        done = terminated or truncated

        # memory.state.append(state)
        # memory.action.append(action.squeeze(0).cpu().numpy())
        # memory.log_prob.append(log_prob.squeeze(0).detach().cpu().numpy())
        # memory.reward.append(reward)
        # memory.value.append(value.item())
        # memory.done.append(done)

        global_memory.state.append(state)
        # global_memory.action.append(action.squeeze(0).cpu().numpy())
        global_memory.action.append(action_env)
        global_memory.log_prob.append(log_prob.squeeze(0).cpu().numpy())
        global_memory.reward.append(reward)
        global_memory.value.append(value.item()) #每步只存V(s_t)
        global_memory.done.append(done)

        state = state_new
        sum_rewards += reward
        global_step += 1
        if done:
            break

    if global_step >= step_update:
        #bootstrap value，只在整个batch结束时加一次
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():
            fea = policy.forward(state_tensor)  # 隐藏层输出
            value_final = policy.value(fea)  # 隐藏层->输出层
        # memory.value.append(value_final)
        global_memory.value.append(value_final.item())

    # #reward标准化
    # reward_np = np.array(memory.reward)
    # reward_mean = np.mean(reward_np)
    # reward_std = np.std(reward_np) + 1e-8
    # reward_normalized = (reward_np - reward_mean) / reward_std
    # advantage, returns = policy.gae(reward_normalized, memory.value, memory.done, lamb, gamma)

        advantage, returns = policy.gae(global_memory.reward, global_memory.value, global_memory.done, lamb, gamma)
        global_memory.advantage = advantage
        global_memory.returns = returns
        ppo(global_memory, batch_size)

        global_memory = Memory()
        global_step = 0

    # advantage, returns = policy.gae(memory.reward, memory.value, memory.done, lamb, gamma)
    # memory.advantage = advantage
    # memory.returns = returns
    # ppo(memory, 64)

    all_rewards.append(sum_rewards)
    print(f"Episode {episode}, total reward: {sum_rewards}")
    # ave_reward = sum_rewards / len(global_memory.reward)
    # print(f"Episode {episode}, total reward: {ave_reward}")

# window = 20
# plt.plot(np.convolve(all_rewards, np.ones(window) / window, mode='valid')) #滑动平滑画reward
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
text = f"lr: {lr_optim}\nbatch_size: {batch_size}"
plt.text(x=500, y=-700, s=text)
plt.show()