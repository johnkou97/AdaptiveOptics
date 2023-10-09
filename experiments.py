# design the experiments
from actor_critic import ActorCritic, HyperparamsActorCritic
from helper import smooth, LearningCurvePlot
import numpy as np
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system

env = Sharpening_AO_system()

n_epochs = 1000
n_runs = 3

plot = LearningCurvePlot()

print("Running experiments for n_steps...")


print("N_Steps: 1")
results = []
for run in range(n_runs):
    print(f"Run {run+1} of {n_runs}")
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=1, batch_size=4, trace_length=100, 
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))
np.save('experiments/n_steps_1.npy', np.array(results))
plot.add_curve(results, label='Actor-Critic (n=1)')

print("N_Steps: 2")
results = []
for run in range(n_runs):
    print(f"Run {run+1} of {n_runs}")
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=2, batch_size=4, trace_length=100,
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))
np.save('experiments/n_steps_2.npy', np.array(results))
plot.add_curve(results, label='Actor-Critic (n=2)')

print("N_Steps: 5")
results = []
for run in range(3):
    print(f"Run {run+1} of {n_runs}")
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=5, batch_size=4, trace_length=100,
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))
np.save('experiments/n_steps_5.npy', np.array(results))
plot.add_curve(results, label='Actor-Critic (n=5)')


plot.save(f'experiments/actor_critic_n_steps.png')

