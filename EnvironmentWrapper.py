import gym
import numpy as np
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
from gym_ao.gym_ao.gym_centering import Centering_AO_system
from gym_ao.gym_ao.gym_sharpening_easy import Sharpening_AO_system_easy
from gym_ao.gym_ao.gym_darkhole import Darkhole_AO_system
import hcipy as hp
import matplotlib.pyplot as plt
import os

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

    def render(self, mode='animation', episode=None, iteration=None, tot_rewards=None, loc='test'):
        if mode == 'animation':
            if not hasattr(self, 'fig'):
                self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 10))
            for ax in self.axes.ravel():
                ax.cla()

            # Plot focal plane image
            plt.sca(self.axes[0, 0])
            plt.axis('off')
            plt.title(f'Image, Strehl: {self.env.strehl*100:.2f}%')
            im1 = hp.imshow_field(self.env.image, cmap='viridis', vmin=0)

            # if self.cbar1 does not exist, create it otherwise, update it
            if hasattr(self, 'cbar1'):
                self.cbar1.update_normal(im1)
            else:
                self.cbar1 = plt.colorbar(im1)

            plt.sca(self.axes[0, 1])
            im2 = hp.imshow_field(np.log10(self.env.image), vmax=0,
                                vmin=-4, cmap='inferno')
            plt.axis('off')
            plt.title(f'log10 Image')
            if hasattr(self, 'cbar2'):
                self.cbar2.update_normal(im2)
            else:
                self.cbar2 = plt.colorbar(im2)

            # Plot mirror shape
            plt.sca(self.axes[1, 0])
            dm_phase = self.env.deformable_mirror.phase_for(self.env.wavelength) * self.env.aperture
            vmax = np.max(np.abs(dm_phase))
            im3 = hp.imshow_field(dm_phase, cmap='bwr', vmin=-vmax, vmax=vmax)
            plt.axis('off')
            plt.title('Deformable mirror shape')
            if hasattr(self, 'cbar3'):
                self.cbar3.update_normal(im3)
            else:
                self.cbar3 = plt.colorbar(im3)
            
            plt.sca(self.axes[1, 1])

            if episode is not None:

                if episode > 0:
                    plt.plot(np.arange(episode), tot_rewards, marker='o',
                            color='black')
                    plt.ylabel('Episode reward')
                    plt.xlabel('Episode')

                plt.suptitle(f'Episode: {episode}, iteration: {iteration}')
            plt.tight_layout()
            if episode is not None:
                if not os.path.exists(f"figures/animations/{loc}"):
                    os.makedirs(f"figures/animations/{loc}")

                plt.savefig(f"figures/animations/{loc}/{episode}_{iteration}.png")
                # plt.savefig(f"figures/animations/{loc}/svg_{episode}_{iteration}.svg")

