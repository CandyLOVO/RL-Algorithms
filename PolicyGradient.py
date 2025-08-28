import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim

from pygame.display import update

env = gymnasium.make("CartPole-v1") #预设环境



for episode in range(500):
    observation = env.reset() #重置环境，初始状态观测空间
    while True:
        #算法
        #状态->动作概率
        action = select_next_action(observation) #算法决定
        ####
        next_observation, reward, terminated, truncated, info = env.step(action) #执行动作，映射环境定义
        done = terminated or truncated
        if done:
            break

        update(observation, action, reward, terminated) #算法更新模型参数
        observation = next_observation #更新观测空间

env.close()

