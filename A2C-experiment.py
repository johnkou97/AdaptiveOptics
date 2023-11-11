from stable_baselines3 import A2C
from EnvironmentWrapper import CustomEnvWrapper
from callbacks import WandbCustomCallback
import wandb

# Set up Weights and Biases
project_name = "centering-ao-system"  # needs to change for each experiment
# options: sharpening-ao-system, sharpening-ao-system-easy, centering-ao-system, darkhole-ao-system

config = {
    "policy_type": "MlpPolicy",
    "env_name": "Centering_AO_system" # needs to change for each experiment corresponding to project_name
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

# Create the SAC model with the custom callback
model = A2C("MlpPolicy", env, verbose=1)

# Create an experiment
n_timesteps = 100000
n_runs = 3

print("Running experiment with A2C...")

group_name = f"A2C-{env.env.wf_rms}rms-{env.action_space.shape[0]}act" 
# needs to change if you use sharpeing-ao-system or darkhole-ao-system with zernike modes to
# indicate the use of zernike modes in the group name
run_num = get_run_num(api.runs(f"adapt_opt/{project_name}"), group_name)

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
    model.learn(total_timesteps=n_timesteps, callback=WandbCustomCallback(), progress_bar=True)
    wandb.finish()
    run_num += 1

# Close the environment
env.close()