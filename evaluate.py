from stable_baselines3 import SAC
from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper
import matplotlib.pyplot as plt
import numpy as np
import tqdm

experiment_name = "centering_ao_system"
model_name = 'SAC-3rms-3act-1000buf-4'

eval_episodes = 1000
eval_steps = 100

# load the model
model = SAC.load(f"models/{model_name}")

# Create the Gym wrapper
env = CustomEnvWrapper(name="Centering_AO_system")

# Evaluate the agent

plt.figure(figsize=(10,5))
bins = np.linspace(-0.025, 0, 100)

print(f"Evaluating agent {model_name} on {experiment_name}...") 
average_reward = []
for episode in tqdm.tqdm(range(eval_episodes)):
    rewards = []
    obs = env.reset()
    for step in range(eval_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        rewards.append(reward)

    # keep track of rewards
    average_reward.append(sum(rewards)/len(rewards))


# histogram of rewards
plt.hist(average_reward, bins=bins, label=f"SAC-{model_name.split('-')[3]}")

# no agent
print(f"Evaluating no agent on {experiment_name}...")
average_reward = []
for episode in tqdm.tqdm(range(eval_episodes)):
    rewards = []
    obs = env.reset()
    for step in range(eval_steps):
        action = 0
        obs, reward, done, info = env.step(action)
        # env.render()
        rewards.append(reward)

    # keep track of rewards
    average_reward.append(sum(rewards)/len(rewards))

# histogram of rewards
plt.hist(average_reward, bins=bins, label="No Agent")

plt.xlabel("Average Reward")
plt.ylabel("Frequency")
plt.xlim(bins[0], bins[-1]) if bins[0] < bins[-1] else plt.xlim(bins[-1], bins[0])
plt.legend()
plt.savefig(f"figures/evaluation_{experiment_name}.png")
plt.close()





    
    
