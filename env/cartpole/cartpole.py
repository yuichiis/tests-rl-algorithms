import gym

env = gym.make('CartPole-v1',render_mode="human")
env.reset()

for _ in range(1000):
    #env.render()
    action = env.action_space.sample()
    observation, reward, done, truncate, info = env.step(action)
    print(done, truncate)
    if done:
        env.reset()
env.close()
