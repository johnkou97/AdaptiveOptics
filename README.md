# Reinforcement Learning for Adaptive Optics


## Table of Contents

- [Table of Contents](#table-of-contents)
- [How to use](#how-to-use)
    - [Installation](#installation)
    - [Structure of the repository](#structure-of-the-repository)
    - [Running the Environments and Training Agents](#running-the-environments-and-training-agents)
    - [Important Notes](#important-notes)
- [Environments](#environments)
    - [Image sharpening](#image-sharpening)
    - [Dark hole](#dark-hole)
    - [Image sharpening easy](#image-sharpening-easy)
    - [Image centering](#image-centering)
- [Results](#results)
    - [Centering](#centering)
    - [Image sharpening easy](#image-sharpening-easy-1)
        - [Evaluation for 2 zernike modes](#evaluation-for-2-zernike-modes)
        - [Evaluation for 5 zernike modes](#evaluation-for-5-zernike-modes)
        - [Evaluation for 9 zernike modes](#evaluation-for-9-zernike-modes)
        - [Evaluation for 14 zernike modes](#evaluation-for-14-zernike-modes)
        - [Evaluation for 20 zernike modes](#evaluation-for-20-zernike-modes)
        - [Evaluation for 27 zernike modes](#evaluation-for-27-zernike-modes)
- [Animations](#animations)
    - [Centering](#centering-1)
        - [Baseline (no agent)](#baseline-no-agent)
        - [Best agent](#best-agent)
    - [Sharpening easy with 20 zernike modes](#sharpening-easy-with-20-zernike-modes)
        - [Baseline (no agent)](#baseline-no-agent-1)
        - [Best agent](#best-agent-1)


## How to use

### Installation

1. Clone the repository
```bash
git clone https://github.com/johnkou97/AdaptiveOptics.git
```

2. Navigate to the directory
```bash
cd AdaptiveOptics
```

3. Also clone the submodule
```bash
git submodule update --init --recursive
```
Sometimes you might need to first delete the submodule folder and then run the command again.
```bash
rm -rf gym_ao/gym_ao
git submodule update --init --recursive
```

4. Create a virtual environment using the provided `environment.yml` file (optional but recommended)
```bash
conda env create -f environment.yml
conda activate adapt
```

5. You are now ready to run the experiments. 

### Structure of the repository

- `A2C-train.py`: Script to train A2C agents.
- `SAC-train.py`: Script to train SAC agents.
- `A2C-experiment.py`: Script to run experiments with A2C agents.
- `SAC-experiment.py`: Script to run experiments with SAC agents.
- `evaluate.py`: Script to evaluate trained models.
- `no_agent.py`: Script to run baseline simulations without an agent.
- `extract_wandb.py`: Script to extract data from Weights & Biases.
- `rendering.py`: Script for visualization and rendering.
- `helper.py`: Contains helper functions for the project.
- `callbacks.py`: Contains custom callbacks for training.
- `EnvironmentWrapper.py`: Contains the environment wrapper classes.

- `gym_ao/`: Contains the gym environments for adaptive optics.
  - `gym_ao/gym_centering.py`: Image centering environment.
  - `gym_ao/gym_darkhole.py`: Dark hole environment.
  - `gym_ao/gym_sharpening.py`: Image sharpening environment.
  - `gym_ao/gym_sharpening_easy.py`: Simplified image sharpening environment.

- `models/`: Contains trained models.
- `figures/`: Contains experiment results and visualizations.

- `LICENSE`: MIT License file.
- `README.md`: Project documentation.
- `STRUCTURE.md`: This file.
- `requirements.txt`: Python dependencies for pip.
- `environment.yml`: Conda environment specification.


### Running the Environments and Training Agents

1. The environments can be run directly to test their functionality:

```bash
# Image sharpening
python gym_ao/gym_ao/gym_sharpening.py

# Dark hole
python gym_ao/gym_ao/gym_darkhole.py

# Image sharpening easy
python gym_ao/gym_ao/gym_sharpening_easy.py

# Image centering
python gym_ao/gym_ao/gym_centering.py
```

2. The repository provides scripts for training A2C and SAC agents:

**Training A2C agents:**
```bash
python A2C-train.py
```

**Training SAC agents:**
```bash
python SAC-train.py
```

You may need to modify the scripts to change hyperparameters or the target environment.

3. Running Experiments

**A2C experiments:**
```bash
python A2C-experiment.py
```

**SAC experiments:**
```bash
python SAC-experiment.py
```

4. To evaluate trained models:

```bash
python evaluate.py
```

You'll need to modify the script to specify which model to load and evaluate.

6. For generating visualizations of agent performance:

```bash
python rendering.py
```

7. If you've used Weights & Biases for tracking experiments, you can extract the data using:

```bash
python extract_wandb.py
```

### Important Notes

1. **Weights & Biases Integration:** The training scripts are set up to use Weights & Biases for experiment tracking. You'll need to modify the project and entity names in the scripts or remove the WandbCustomCallback if you don't want to use it.

2. **Model Parameters:** The naming convention for saved models includes information about the training parameters:
   - Algorithm (A2C or SAC)
   - RMS value for aberrations (e.g., 1.7rms)
   - Number of actuators (e.g., 10act)
   - Buffer size for SAC (e.g., 10000buf)
   - Run number (e.g., -0, -1, etc.)

3. **Hardware Requirements:** Training agents, especially with many actuators, can be computationally intensive. All our experiments were run with CPU only, but GPU acceleration is recommended for faster training.

4. **Customizing Environments:** You can modify the environment parameters in the gym_ao implementations to adjust difficulty, observation space, and other settings.


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

### Image sharpening easy

The goal of this environment is to maximize the Strehl ratio based on focal plane images like in the image sharpening environment. The difference is that the aberrations from the atmosphere can always be corrected by the zernike modes of the deformable mirror. 

- Observation: The observed (noisy) image intensity in the focal plane. The image is normalized such that the values are always between 0 and 1. The image has a size of 96x96 pixels.
- Action: An array of commands to send to the actuators to reshape the deformable mirror. This is in units of radians and should have an absolute value smaller than 0.3 to avoid divergence. Default is with 20 modes of zernike.
- Reward: The Strehl ratio, which is a measure of image sharpness and is between 0 and 1.

Run the environment with the following command:

```python gym_ao/gym_ao/gym_sharpening_easy.py```

### Image centering

The goal of this environment is to minimize the distance between the center of the image and the center of the focal plane.

- Observation: The observed (noisy) image intensity in the focal plane. The image is normalized such that the values are always between 0 and 1. The image has a size of 96x96 pixels.
- Action: An array of commands to send to the actuators to reshape the deformable mirror. This is in units of radians and should have an absolute value smaller than 0.3 to avoid divergence. For this environment we only use 2 zernike modes (tip and tilt) to correct the aberrations.
- Reward: The negative of the distance between the center of the image and the center of the focal plane.

Run the environment with the following command:

```python gym_ao/gym_ao/gym_centering.py```


## Results

We use Weights & Biases to track the results of the experiments. 

### Centering

Find the training results [here](https://api.wandb.ai/links/adapt_opt/gbkd3qfs).

#### Evaluation

We evaluate each agent on 1000 episodes. Each episode is 100 steps long.

![](figures/evaluation_centering_ao_system.png)

### Sharpening easy

We have trained agents on the sharpening easy environment with 2 and 5 and 9 zernike modes.
Reminder that for this environment the aberrations can always be corrected by the zernike modes of the deformable mirror that we use each time.

Find the training results [here](https://api.wandb.ai/links/adapt_opt/5y122g06).

#### Evaluation for 2 zernike modes

We evaluate each agent on 1000 episodes. Each episode is 100 steps long. 

![](figures/evaluation_Sharpening_AO_system_easy.png)

We also evaluate the best performing agent on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-2.png)

#### Evaluation for 5 zernike modes

We evaluate each agent on 1000 episodes. Each episode is 100 steps long.

![](figures/evaluation_Sharpening_AO_system_easy-6act.png)

We also evaluate the best performing agent on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-6act-2.png)

#### Evaluation for 9 zernike modes

We evaluate each agent on 1000 episodes. Each episode is 100 steps long.

![](figures/evaluation_Sharpening_AO_system_easy-10act.png)

We also evaluate the best performing agent on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-10act-2.png)

#### Evaluation for 14 zernike modes

We evaluate each agent on 1000 episodes. Each episode is 100 steps long.

![](figures/evaluation_Sharpening_AO_system_easy-15act.png)

We also evaluate the best performing agent on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-15act-2.png)

#### Evaluation for 20 zernike modes

We evaluate each agent on 1000 episodes. Each episode is 100 steps long.

![](figures/evaluation_Sharpening_AO_system_easy-21act.png)

We also evaluate the best performing agent on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-21act-2.png)

#### Evaluation for 27 zernike modes

Since we only trained one agent for 27 zernike modes, we evaluate it on 10000 episodes of 100 steps and compare it to the performance of the baseline.

![](figures/evaluation_Sharpening_AO_system_easy-28act.png)


## Animations

### Centering

#### Baseline (no agent)

![](figures/animations/no_agent_Centering.gif)

#### Best agent

Our best agent is the `SAC-3rms-3act-1000buf-3`.

![](figures/animations/SAC-3rms-3act-1000buf-3.gif)

### Sharpening easy with 20 zernike modes

#### Baseline (no agent)

![](figures/animations/no_agent.gif)

#### Best agent

Our best agent is the `SAC-1.7rms-21act-100000buf-2`.

![](figures/animations/SAC-1.7rms-21act-100000buf-2.gif)
