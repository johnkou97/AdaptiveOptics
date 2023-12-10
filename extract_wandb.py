import numpy as np
from helper import LearningCurvePlot, smooth
import wandb
import os

# check if the reward_data folder exists
if not os.path.exists('reward_data'):
    os.makedirs('reward_data')

# check if the figures folder exists
if not os.path.exists('figures/learning_curves'):
    os.makedirs('figures/learning_curves')

api = wandb.Api()

def plot_learning_curves(groups, length, ylim=(0, 1), name='test', types='png', dpi=400, project_name='sharpening-ao-system-easy'):
    runs = api.runs(f"adapt_opt/{project_name}")
    plot = LearningCurvePlot(y_lim=ylim, length=length)
    for group in groups:
        print(f'Working on {group}')
        rewards = []
        for run in runs:
            if group in run.name:
                
                # check if it exists locally
                reward_file = f'reward_data/{run.name}.npy'
                try:
                    r = np.load(reward_file)
                    if len(r) < length-1:
                        print(f'WARNING: {reward_file} has only {len(r)} entries, but {length} are required')
                    else:
                        rewards.append(smooth(r[:length-1], length//20))
                except:
                    print(f'Fetching {run.name}')
                    # fetch the logged data for this run
                    run_data = run.scan_history()
                    # get the reward data
                    reward_data = [x["reward"] for x in run_data]
                    # save the reward data
                    np.save(reward_file, reward_data)
                    rewards.append(smooth(reward_data[:length-1], length//20))

        plot.add_curve(rewards, f'{group.split("-")[0]}-{group.split("-")[-1]}') if group.split("-")[0] == 'SAC' else plot.add_curve(rewards, f'{group.split("-")[0]}')
        
    plot.save(f'figures/learning_curves/{name}.{types}', dpi=dpi)


if __name__ == '__main__':


    project_name = 'centering-ao-system'
    # define names of the groups to plot
    groups = ['SAC-3rms-3act-1000buf', 'SAC-3rms-3act-100buf', 'SAC-3rms-3act-10buf', 'A2C-3rms-3act', 'no_agent-3rms']
    length = 100000

    plot_learning_curves(groups, length, ylim=(-0.1,0) ,name='centering', types='png', project_name=project_name, dpi=900)

    project_name = 'sharpening-ao-system-easy'
    # define names of the groups to plot
    groups = ['SAC-1.7rms-3act-1000buf', 'SAC-1.7rms-3act-10000buf', 'SAC-1.7rms-3act-20000buf','SAC-1.7rms-3act-100buf', 'A2C-1.7rms-3act', 'no_agent-1.7rms-3act']
    length = 100000

    plot_learning_curves(groups, length, name='easy_2zer', types='png', project_name=project_name, dpi=900)


    # define names of the groups to plot
    groups = ['SAC-1.7rms-6act-1000buf', 'SAC-1.7rms-6act-10000buf', 'SAC-1.7rms-6act-20000buf', 'A2C-1.7rms-6act', 'no_agent-1.7rms-6act']
    length = 200000

    plot_learning_curves(groups, length, name='easy_5zer', types='png', project_name=project_name, dpi=900)


    # define names of the groups to plot
    groups = ['SAC-1.7rms-10act-10000buf', 'SAC-1.7rms-10act-20000buf', 'A2C-1.7rms-10act', 'no_agent-1.7rms-10act']
    length = 200000

    plot_learning_curves(groups, length, name='easy_9zer', types='png', project_name=project_name, dpi=900)


    # define names of the groups to plot
    groups = ['SAC-1.7rms-15act-10000buf', 'SAC-1.7rms-15act-20000buf', 'SAC-1.7rms-15act-50000buf', 'A2C-1.7rms-15act', 'no_agent-1.7rms-15act']
    length = 300000

    plot_learning_curves(groups, length, name='easy_14zer', types='png', project_name=project_name, dpi=900)


    # define names of the groups to plot
    groups = ['SAC-1.7rms-21act-20000buf', 'SAC-1.7rms-21act-50000buf', 'SAC-1.7rms-21act-100000buf', 'A2C-1.7rms-21act', 'no_agent-1.7rms-21act']
    length = 800000

    plot_learning_curves(groups, length, name='easy_20zer', types='png', project_name=project_name, dpi=900)

    # define names of the groups to plot
    groups = ['SAC-1.7rms-28act-100000buf', 'no_agent-1.7rms-28act']
    length = 2000000

    plot_learning_curves(groups, length, name='easy_27zer', types='png', project_name=project_name, dpi=900)