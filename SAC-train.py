import os
import gym
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
import wandb
from callbacks import WandbCustomCallback

# Set up Weights and Biases

config = {
    "policy_type": "MlpPolicy",
    "env_name": "Sharpening_AO_system"
}

api = wandb.Api()

runs = api.runs("adapt_opt/sharpening-ao-system")

group_name = "SAC-test"

run_num = 0
for run in runs:
    if group_name in run.name:
        run_num += 1

run = wandb.init(
    group=group_name,
    name=f"SAC-test-run-{run_num}",
    project="sharpening-ao-system",
    entity="adapt_opt",
    config=config,
    sync_tensorboard=True,
)


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

# Create and train the SAC model and sync with wandb
model = SAC("MlpPolicy", env, verbose=1, buffer_size=10)
model.learn(total_timesteps=110, callback=WandbCustomCallback(), progress_bar=True)
model.save("sac_sharpening_ao_system")

# Close the environment
env.close()

