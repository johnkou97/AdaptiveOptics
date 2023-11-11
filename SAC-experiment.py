from stable_baselines3 import SAC
from callbacks import WandbCustomCallback
from EnvironmentWrapper import CustomEnvWrapper
import wandb

# Set up Weights and Biases
project_name = "sharpening-ao-system"  # needs to change for each experiment
# options: sharpening-ao-system, sharpening-ao-system-easy, centering-ao-system, darkhole-ao-system

config = {
    "policy_type": "MlpPolicy",
    "env_name": "Sharpening_AO_system" # needs to change for each experiment corresponding to project_name
    # options: Sharpening_AO_system, Sharpening_AO_system_easy, Centering_AO_system, Darkhole_AO_system
}

api = wandb.Api()

def get_run_num(runs, group_name):
    run_num = 0
    for run in runs:
        if group_name in run.name:
            run_num += 1
    return run_num

# Create the Gym wrapper
env = CustomEnvWrapper(name=config["env_name"])

# Create an experiment
n_timesteps = 100000
n_runs = 3
buffer_size = 1000

print("Running experiment with SAC...")

group_name = f"SAC-{env.env.wf_rms}rms-{env.action_space.shape[0]}act-{buffer_size}buf"
# needs to change if you use sharpeing-ao-system or darkhole-ao-system with zernike modes to
# indicate the use of zernike modes in the group name
run_num = get_run_num(api.runs("adapt_opt/sharpening-ao-system"), group_name)

for run in range(n_runs):
    print(f"Run {run+1} of {n_runs}")
    run = wandb.init(
        group=group_name,
        name=f"{group_name}-{run_num}",
        project=project_name,
        entity="adapt_opt",
        config=config,
        sync_tensorboard=True,
    )
    env.reset()
    model = SAC(config["policy_type"], env, verbose=1, buffer_size=buffer_size)
    model.learn(total_timesteps=n_timesteps, callback=WandbCustomCallback(), progress_bar=True)
    wandb.finish()
    run_num += 1

# Close the environment
env.close()