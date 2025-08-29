import gymnasium
import torch
#神经网络类式接口
import torch.nn as nn
#神经网络函数接口
import torch.nn.functional as F
import matplotlib.pyplot as plt

#状态->动作概率
class Policy(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, action_dim=2): #隐藏层神经元数量暂定36 4*9
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    #在下方训练中传入当前环境参数
    def forward(self, x): #x必须是张量
        x = F.relu(self.fc1(x)) #输入映射ReUL
        # x = F.relu(self.fc2(x))
        # output = F.softmax(self.fc3(x), dim=-1) #dim=-1直接选中特征数，无需记忆索引
        output = F.softmax(self.fc2(x), dim=-1)
        return output #动作概率分布

env = gymnasium.make("CartPole-v1")
# env = gymnasium.make("CartPole-v1", render_mode="human") #预设环境
policy = Policy()
lr_optim = 0.001
optimizer = torch.optim.Adam(policy.parameters(), lr=lr_optim)

gamma = 0.99 #折扣因子γ
all_rewards = []
for episode in range(500): #episode一局
    state, info = env.reset() #初始化环境[位置，速度，角度，角速度]
    log_prob = [] #logΠ(a|s)
    rewards = []
    sampled_action = []

    while True:
    #算法
        state_tensor = torch.tensor(state, dtype=torch.float) #转化为张量，PyTorch定义的神经网络只接收张量输入
        action_prob = policy.forward(state_tensor) #当前环境参数映射得概率，离散

        sampled_action = torch.multinomial(action_prob,1) #采样，返回索引
        log_prob.append(torch.log(action_prob[sampled_action])) #（取出概率值）取对数！！！！->放到log_prob列表最后(.append)

        action = sampled_action.item() #索引转化成整数
        state, reward, terminated, truncated, info = env.step(action) #环境执行，动作索引->新状态，及时奖励，自然终止，截断，额外info
        rewards.append(reward)  #更新reward

        done = terminated or truncated #两个终止标志合并
        if done: break

    #回合结束，策略梯度算法更新θ
    returns = []
    G_t = 0
    loss = 0
    for r in rewards[::-1]: #反向遍历
        G_t = r + gamma * G_t
        returns.insert(0, G_t) #恢复原顺序
    returns = torch.tensor(returns) #转化为张量（同上）
    returns = (returns - returns.mean()) / (returns.std() + 1e-5)  #去基线b/奖励标准差
    for log_p, G in zip(log_prob, returns): #取变量赋值
        loss = loss - G * log_p
    loss = torch.mean(loss) #要算损失的均值，否则受episode数值影响

    optimizer.zero_grad() #梯度清零
    loss.backward() #反向传播，自动计算loss对模型所有可训练参数的梯度
    optimizer.step() #更新θ
    ####

    total_reward = sum(rewards)
    all_rewards.append(total_reward)
    if episode % 50 == 0:
        print(f"Episode {episode}, total reward: {total_reward}")

plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total reward")
text = f"lr: {lr_optim}\ngamma: {gamma}"
plt.text(x=50, y=400, s=text)
plt.show()
