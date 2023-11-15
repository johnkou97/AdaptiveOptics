from stable_baselines3 import SAC
from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper


# load the model
model = SAC.load("models/SAC-3rms-3act-1000buf-3")

# Create the Gym wrapper
env = CustomEnvWrapper(name="Centering_AO_system")

# Evaluate the agent

for episode in range(10):
    rewards = []
    obs = env.reset()
    for step in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        # print(f"Reward: {rewards}")
        # store the reward
        rewards.append(reward)

    # keep track of rewards
    print(f"Episode: {episode+1}")
    print(f"Total reward: {sum(rewards)}")
    print(f"Average reward: {sum(rewards)/len(rewards)}")
    print(f"Final reward: {rewards[-1]}")


    
    
