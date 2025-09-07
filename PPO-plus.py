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
        self.rewards = []
        self.log_prob = []
        self.values = []
        self.done = []
        self.advantage = []
        self.returns = []

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        # self.std = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Parameter(torch.zeros(action_dim)) #网络只学习均值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        mean = a_max * torch.tanh(mean) #限制在[-2,2]
        # std = self.std(x)
        # std = F.softplus(std) #保证std是正数
        std = self.std.exp()
        #限制大小后的均值，标准差——网络训练预测数据
        return mean, std

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
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

class PPO_Agent:
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, lam, actor_lr, critic_lr, batch_size, clip, c1, c2):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim) #神经网络
        self.critic = CriticNet(state_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), actor_lr) #优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), critic_lr)
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.clip = clip
        self.c1 = c1
        self.c2 = c2

    def get_action(self, state):
        mean, std = self.actor.forward(state)
        std = torch.clamp(std, min=1e-6) #限制std最小值
        dist = torch.distributions.Normal(mean, std) #分布
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) #对数动作概率——log_prob_old
        #抽到得到的动作，动作对应的对数动作概率
        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.actor.forward(state)
        std = torch.clamp(std, min=1e-6)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) #原有状态动作，在新分布下的对数动作概率——log_prob_new
        entropy = dist.entropy().sum(-1, keepdim=True)
        #新分布的对数动作概率，熵值
        return log_prob, entropy

    def gae(self, reward, value, done):
        reward = np.array(reward, dtype=np.float32)
        value = np.array(value, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        T = len(reward)
        advantage = np.zeros_like(reward, dtype=np.float32)
        ad = 0.0
        for t in reversed(range(T)):
            delta = reward[t] + self.gamma * value[t+1] * (1 - done[t]) - value[t]
            ad = delta + self.gamma * self.lam * ad * (1 - done[t])
            advantage[t] = ad
        returns = advantage + value[:-1]
        advantage = (advantage - advantage.mean()) / advantage.std() #优势函数标准化
        #GAE算法输出值（目标值），优势函数值
        return torch.tensor(returns, dtype=torch.float32), torch.tensor(advantage, dtype=torch.float32)

    def ppo_update(self, state, action, log_prob_old, returns, advantage):
        state = torch.from_numpy(np.asarray(state, dtype=np.float32))
        action = torch.from_numpy(np.asarray(action, dtype=np.float32))
        log_prob_old = torch.from_numpy(np.asarray(log_prob_old, dtype=np.float32)).reshape(-1, 1)
        returns = torch.from_numpy(np.asarray(returns, dtype=np.float32)).reshape(-1, 1)
        advantage = torch.from_numpy(np.asarray(advantage, dtype=np.float32)).reshape(-1, 1)

        N = len(state)
        for _ in range(5):
            indices = torch.randperm(N)
            state = state[indices]
            action = action[indices]
            log_prob_old = log_prob_old[indices]
            returns = returns[indices]
            advantage = advantage[indices]

            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                state_batch = state[start:end]
                action_batch = action[start:end]
                log_prob_old_batch = log_prob_old[start:end].detach()
                returns_batch = returns[start:end].detach()
                advantage_batch = advantage[start:end].detach()

                log_prob_new, entropy = self.evaluate(state_batch, action_batch)
                ratio = torch.exp(log_prob_new - log_prob_old_batch)
                l1 = ratio * advantage_batch
                l2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantage_batch
                l_clip = -torch.mean(torch.min(l1, l2))
                l_entropy = -torch.mean(entropy)
                actor_loss = l_clip + self.c2 * l_entropy #策略参数梯度贡献

                value = self.critic.forward(state_batch)
                l_value = torch.mean((value.squeeze() - returns_batch)**2)
                critic_loss = self.c1 * l_value #值参数梯度贡献

                #梯度更新
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

#---------------------------------------------主函数---------------------------------------------#
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 128
gamma = 0.99
lam = 0.95
actor_lr = 3e-4
critic_lr = 1e-3
batch_size = 64
clip = 0.2
c1 = 0.5
c2 = 0.01
memory = Memory()
ppo = PPO_Agent(state_dim, hidden_dim, action_dim, gamma, lam, actor_lr, critic_lr, batch_size, clip, c1, c2)
steps = 0
step_update = 2048
reward_all = []

for episode in range(2000):
    state, info = env.reset()
    reward_sum = 0

    #200步内状态收集更新
    for t in range(200):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = ppo.get_action(state_tensor)
            value = ppo.critic.forward(state_tensor)

        action_env = action.detach().cpu().numpy().reshape(-1)
        state_new, reward, terminated, truncated, info = env.step(action_env)
        reward_sum += reward
        reward = (float(reward) + 8.0) / 8.0
        done = terminated or truncated

        memory.state.append(state)
        memory.action.append(action_env)
        memory.log_prob.append(log_prob.squeeze(0).cpu().numpy())
        memory.rewards.append(reward)
        memory.values.append(value.item())
        memory.done.append(done)

        state = state_new
        steps += 1
        if done:
            break

    #结束一轮，batch数据收集够了，需要计算advantage、returns，并用ppo函数更新策略
    if steps >= step_update:
        # 补上value的第 T+1 位值
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value_final = ppo.critic.forward(state_tensor)
        memory.values.append(value_final.item())

        #reward标准化 -> 计算value目标值（GAE近似）
        reward_np = np.array(memory.rewards)
        reward_mean = np.mean(reward_np)
        reward_std = np.std(reward_np) + 1e-8
        reward_normalized = (reward_np - reward_mean) / reward_std
        returns, advantage = ppo.gae(reward_normalized, memory.values, memory.done)
        memory.returns = returns
        memory.advantage = advantage

        ppo.ppo_update(memory.state, memory.action, memory.log_prob, memory.returns, memory.advantage)

        memory = Memory() #重置memory
        steps = 0

    reward_all.append(reward_sum)
    print(f"Episode {episode}, total reward: {reward_sum}")

plt.plot(reward_all)
plt.xlabel('Episode')
plt.ylabel('Reward')
text = f"actor_lr: {actor_lr}\ncritic_lr: {critic_lr}\nbatch_size: {batch_size}"
plt.text(x=0, y=-400, s=text)
plt.show()