import gym
import numpy as np
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
from gym_ao.gym_ao.gym_centering import Centering_AO_system
from gym_ao.gym_ao.gym_sharpening_easy import Sharpening_AO_system_easy
from gym_ao.gym_ao.gym_darkhole import Darkhole_AO_system

class CustomEnvWrapper(gym.Env):
    def __init__(self, name):
        
        if name == "Sharpening_AO_system":
            self.env = Sharpening_AO_system()
            self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(self.env.num_modes,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

        elif name == "Sharpening_AO_system_easy":
            self.env = Sharpening_AO_system_easy()
            self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(self.env.num_modes,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

        elif name == "Centering_AO_system":
            self.env = Centering_AO_system()
            self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(self.env.num_modes,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

        elif name == "Darkhole_AO_system": ## needs fixing for observation space
            self.env = Darkhole_AO_system()
            self.action_space = gym.spaces.Box(low=-0.3, high=0.3, shape=(self.env.num_modes,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=0, high=1., shape=self.env.observation_space.shape, dtype=np.float32)

        else:
            raise ValueError("Invalid environment name: ", self.name)

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


