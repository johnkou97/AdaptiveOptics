import os
import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
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
        self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(4,), dtype=np.float32)
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

# Create the SAC model with the custom callback
model = SAC("MlpPolicy", env, verbose=1, buffer_size=10000)
callback = RewardCallback()

# Create an experiment
n_timesteps = 100000
n_runs = 3

plot = LearningCurvePlot()

print("Running experiment with SAC...")

results = []
for run in range(n_runs):
    print(f"Run {run+1} of {n_runs}")
    model.learn(total_timesteps=n_timesteps, callback=callback, progress_bar=True)
    rewards = callback.rewards
    rewards_flat = [item for sublist in rewards for item in sublist]
    np.save(f'experiments/sac_run_{run}.npy', rewards_flat)
    results.append(smooth(rewards_flat, 51))
    # Reset the callback
    callback.reset()
    print(np.shape(results))
plot.add_curve(results, label='SAC')

plot.save("sac_sharpening_experiment.png")

# Close the environment
env.close()