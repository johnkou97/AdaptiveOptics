# design the experiments
from actor_critic import ActorCritic, HyperparamsActorCritic
from helper import smooth, LearningCurvePlot
import numpy as np
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system

env = Sharpening_AO_system()

plot = LearningCurvePlot()

results = []
for run in range(3):
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=2, batch_size=4, trace_length=100, 
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))

plot.add_curve(results, label='Actor-Critic (n=2)') 


results = []
for run in range(3):
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=5, batch_size=4, trace_length=100,
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))

plot.add_curve(results, label='Actor-Critic (n=5)')

results = []
for run in range(3):
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=10, batch_size=4, trace_length=100,
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))

plot.add_curve(results, label='Actor-Critic (n=10)')


plot.save(f'experiments/actor_critic_n_steps.png')

