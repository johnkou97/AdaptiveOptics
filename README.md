# Reinforcement Learning for Adaptive Optics

## Environments

### Image sharpening

The goal of this environment is to maximize the Strehl ratio based on focal plane images. 

- Observation: The observed (noisy) image intensity in the focal plane. The image is normalized such that the values are always between 0 and 1. The image has a size of 96x96 pixels.
- Action: An array of commands to send to the actuators to reshape the deformable mirror. This is in units of radians and should have an absolute value smaller than 0.3 to avoid divergence. Default is 400 actuators.
- Reward: The Strehl ratio, which is a measure of image sharpness and is between 0 and 1.
- Things to consider: 
    * Partially observable Markov decision process: twin image problem  (image intensity and not the electric field)
    * Possible solution: provide the agent a history of observations and commands or through the use of agents that have intrinsic memory

Run the environment with the following command:

```python gym_ao/gym_ao/gym_sharpening.py```

### Dark hole 

The goal of this environment is to remove starlight from a small region if the image. 

- Observation: A measurement with information about the electric field in the dark hole region. The shape is N_probes x N_pixels, default is 5 x 499.
- Action: An array of commands to send to the actuators to reshape the deformable mirror. This is in units of radians and should have an absolute value smaller than 0.3 to avoid divergence. Default is 400 actuators.
- Reward: The log of the contrast (mean of the image intensity in the dark hole region divided by the peak intensity of the starlight).

Run the environment with the following command:

```python gym_ao/gym_ao/gym_darkhole.py```

## Discussion during previous meeting

1. Simplify the environment
2. Try different algorithms
3. Try longer training
4. Large action space
5. Rewards too sparse
6. Use recurrent network
7. Exploration vs exploitation

## What we tried

- We addressed problems 2 and 3 by using `SAC` and `A2C` from `stable-baselines3` and training for 100k steps.

<img src="sac_sharpening.png" width="400" height="200">

- We addressed problem 4 and 1 by using a smaller action space (4 actuators instead of 400) and reducing the RMS of the noise in the observation (from 1.7 to 1.2).

<img src="sac_sharpening_experiment.png" width="400" height="200">

<img src="a2c_sharpening_experiment.png" width="400" height="200">

<img src="a2c_sharpening_experiment_explained.png" width="400" height="200">









