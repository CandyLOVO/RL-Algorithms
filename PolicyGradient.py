import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human")
for i_episode in range(20):
    # reset 现在返回 (observation, info)
    observation, info = env.reset()
    for t in range(100):  # 每局最多 100 步
        # 渲染画面
        env.render()
        # 随机选择一个动作（左推 or 右推）
        action = env.action_space.sample()
        # step 现在返回 5 个值
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # 游戏是否结束

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()