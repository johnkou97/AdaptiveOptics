from stable_baselines3 import SAC, A2C
from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper
import matplotlib.pyplot as plt
import numpy as np
import tqdm

experiment_name = "Sharpening_AO_system_easy"
model_names = ['A2C-1.7rms-3act-3', 'SAC-1.7rms-3act-1000buf-3', 'SAC-1.7rms-3act-10000buf-3', 'SAC-1.7rms-3act-20000buf-6']

eval_episodes = 10000
eval_steps = 100

# Create the Gym wrapper
env = CustomEnvWrapper(name=experiment_name)

# Evaluate the agent

plt.figure(figsize=(10,5))
bins = np.linspace(0, 1, 100)

for model_name in model_names:
    # load the model
    if "SAC" in model_name:
        model = SAC.load(f"models/{model_name}")
    elif "A2C" in model_name:
        model = A2C.load(f"models/{model_name}")
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
    plt.hist(average_reward, bins=bins, label=f"{model_name}", alpha=0.5)

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
plt.hist(average_reward, bins=bins, label="No Agent", alpha=0.5)

plt.xlabel("Average Reward")
plt.ylabel("Frequency")
plt.xlim(bins[0], bins[-1]) if bins[0] < bins[-1] else plt.xlim(bins[-1], bins[0])
plt.legend()
plt.savefig(f"figures/evaluation_{experiment_name}.png")
plt.close()





    
    
