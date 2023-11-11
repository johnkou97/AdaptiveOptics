import numpy as np
from EnvironmentWrapper import CustomEnvWrapper
from helper import smooth
import wandb
import tqdm

# Set up wandb

project_name = "sharpening-ao-system-easy"  # needs to change for each experiment
# options: sharpening-ao-system, sharpening-ao-system-easy, centering-ao-system, darkhole-ao-system

config = {
    "env_name": "Sharpening_AO_system_easy", # needs to change for each experiment corresponding to project_name
    # options: Sharpening_AO_system, Sharpening_AO_system_easy, Centering_AO_system, Darkhole_AO_system
}

api = wandb.Api()

def get_run_num(runs, group_name):
    run_num = 0
    for run in runs:
        if group_name in run.name:
            run_num += 1
    return run_num

print("Testing the environment with no agent")
# run the environment with no actions

env = CustomEnvWrapper(name=config["env_name"])
group_name = f"no_agent-{env.env.wf_rms}rms"
run_num = get_run_num(api.runs("adapt_opt/sharpening-ao-system-easy"), group_name)
run = wandb.init(
    group=group_name,
    name=f"{group_name}-{run_num}",
    project=project_name,
    entity="adapt_opt",
    config=config,
    sync_tensorboard=True
)
env.reset()
rewards = []
for _ in tqdm.tqdm(range(100000)):
    action = np.zeros(env.action_space.shape)
    observation, reward, done, info = env.step(action)
    rewards.append(reward)
    wandb.log({"reward": reward})
env.close()
wandb.finish()

# get the average reward and the standard deviation
rewards = np.array(rewards)
print("Average reward: ", np.mean(rewards))
print("Standard deviation: ", np.std(rewards))
print("Min/Max reward: ", np.min(rewards), "/", np.max(rewards))