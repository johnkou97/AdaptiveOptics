import os
import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system

class CustomEnvWrapper(gym.Env):
    def __init__(self):
        # Initialize your Sharpening_AO_system environment here
        self.env = Sharpening_AO_system()
        self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(400,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render()

# Create the Gym wrapper
env = CustomEnvWrapper()

# Create and train the SAC model
model = SAC("MlpPolicy", env, verbose=1, buffer_size=10000)
model.learn(total_timesteps=100000)  # Adjust the number of timesteps as needed

# Save the trained model
model.save("sac_sharpening")

# Load the trained model if needed
# model = SAC.load("sac_sharpening")

# Test the trained model
obs = env.reset()
total_reward = []
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    total_reward.append(rewards)
    print(f'It: {i+1}, Reward: {rewards}')

# Close the environment
env.close()
