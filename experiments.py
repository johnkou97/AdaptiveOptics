# design the experiments
from actor_critic import ActorCritic, HyperparamsActorCritic
from helper import smooth, LearningCurvePlot
import numpy as np
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system

env = Sharpening_AO_system()

plot = LearningCurvePlot()
results = []
for run in range(5):
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=1, batch_size=4,
                                            bootstrap=True, baseline=True)
    ac = ActorCritic(env, hyperparams_ac, experiment_name=f"actor_critic_run_{run}")
    rewards = ac.train(n_epochs=1000)
    results.append(smooth(rewards, 51))

plot.add_curve(results, label='Actor-Critic')
plot.save(f'experiment_actor_critic.png')

