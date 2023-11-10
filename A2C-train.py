import os
import gym
import numpy as np
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from helper import LearningCurvePlot, smooth
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self):
        # Append the reward to the list after each step
        self.rewards.append(self.locals["rewards"])
        return True
    def reset(self):
        self.rewards = []


class CustomEnvWrapper(gym.Env):
    def __init__(self):
        # Initialize your Sharpening_AO_system environment here
        self.env = Sharpening_AO_system()
        self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(400,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

    def step(self, action):
        observation, reward, done, trunc, info = self.env.step(action)
        if done:
            observation = self.reset()
        if trunc:
            observation = self.reset()
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render()

# Create the Gym wrapper
env = CustomEnvWrapper()

# Create and train the TD3 model with the custom callback
model = A2C("MlpPolicy", env, verbose=1)
callback = RewardCallback()
model.learn(total_timesteps=10000000, callback=callback, progress_bar=True)

# Save the trained model
model.save("a2c_sharpening")

# Get the rewards
rewards = callback.rewards
# Flatten the list of reward arrays into a single list
rewards_flat = [item for sublist in rewards for item in sublist]
# Save the rewards array to a file
np.save("rewards_a2c_sharpening.npy", rewards_flat)

# Plot the rewards
plot = LearningCurvePlot()
plot.add_curve(smooth([rewards_flat], 101), "A2C")
plot.save("a2c_sharpening.png")

# Close the environment
env.close()
