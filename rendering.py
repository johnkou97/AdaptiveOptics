from EnvironmentWrapper import CustomEnvWrapper
import tqdm
from stable_baselines3 import SAC
import imageio
import os

experiment_name = "Sharpening_AO_system_easy"


eval_episodes = 10
eval_steps = 100

# evaluate no agent

env = CustomEnvWrapper(name=experiment_name)

average_reward = []
obs = env.reset()
for episode in tqdm.tqdm(range(eval_episodes)):
    rewards = []
    obs = env.reset()
    for step in range(eval_steps):
        action = 0
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        env.render(episode=episode, iteration = step, tot_rewards = average_reward, loc='no_agent')
    

    # keep track of rewards
    average_reward.append(sum(rewards)/len(rewards))

# evaluate agent
model_name = 'SAC-1.7rms-21act-100000buf-2'
model = SAC.load(f"models/{model_name}")
average_reward = []
for episode in tqdm.tqdm(range(eval_episodes)):
    rewards = []
    obs = env.reset()
    for step in range(eval_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        env.render(episode=episode, iteration = step, tot_rewards = average_reward, loc=f'{model_name}')
    

    # keep track of rewards
    average_reward.append(sum(rewards)/len(rewards))

# make gifs
images = []
filenames = os.listdir('figures/animations/no_agent')
filenames.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

for filename in filenames:
    if filename.startswith('3') or filename.startswith('4'):
        images.append(imageio.imread('figures/animations/no_agent/' + filename))
imageio.mimsave('figures/animations/no_agent.gif', images, duration=1000/5)

# make a gif for the agent
images = []
filenames = os.listdir(f'figures/animations/{model_name}')
filenames.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

for filename in filenames:
    if filename.startswith('3') or filename.startswith('5'):
        images.append(imageio.imread(f'figures/animations/{model_name}/' + filename))
imageio.mimsave(f'figures/animations/{model_name}.gif', images, duration=1000/5)

