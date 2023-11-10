import gym
import numpy as np
from stable_baselines3 import SAC
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
from callbacks import WandbCustomCallback
import wandb


# Set up Weights and Biases

config = {
    "policy_type": "MlpPolicy",
    "env_name": "Sharpening_AO_system"
}

api = wandb.Api()

def get_run_num(runs, group_name):
    run_num = 0
    for run in runs:
        if group_name in run.name:
            run_num += 1
    return run_num

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
model = SAC("MlpPolicy", env, verbose=1, buffer_size=1000)

# Create an experiment
n_timesteps = 100000
n_runs = 3


print("Running experiment with SAC...")

group_name = f"SAC-2act-0.4rms-buff1000"
run_num = get_run_num(api.runs("adapt_opt/sharpening-ao-system"), group_name)

for run in range(n_runs):
    print(f"Run {run+1} of {n_runs}")
    run = wandb.init(
        group=group_name,
        name=f"{group_name}-{run_num}",
        project="sharpening-ao-system",
        entity="adapt_opt",
        config=config,
        sync_tensorboard=True,
    )
    model.learn(total_timesteps=n_timesteps, callback=WandbCustomCallback(), progress_bar=True)
    wandb.finish()
    run_num += 1
