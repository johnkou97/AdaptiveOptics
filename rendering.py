import os
import glob
import tqdm
import imageio.v2 as imageio
import cv2

def Evaluate(env, eval_episodes: int, eval_steps: int, agent= None, agent_name: str = 'no_agent'):
    '''
    Evaluate the environment
    Inputs:
    env: gym environment
    eval_episodes: number of episodes to evaluate
    # eval_steps: number of steps to evaluate
    Outputs:
    average_reward: average reward per episode
    '''
    average_reward = []
    _ = env.reset()
    for episode in tqdm.tqdm(range(eval_episodes)):
        rewards = []
        obs = env.reset()
        for step in range(eval_steps):
            action = 0
            if agent:
                action, _states = agent.predict(obs, deterministic=True)    # get the action from the agent
            obs, reward, _done, _info = env.step(action)                    # take the action
            rewards.append(reward)
            env.render(episode=episode, iteration = step, tot_rewards = average_reward, loc=agent_name)
        # keep track of rewards
        average_reward.append(sum(rewards)/len(rewards))
    return average_reward

def MakeGif(name: str, fps: int = 5, episodes: list = [0, 1]):
    '''
    Create a gif from a series of images in a folder
    Inputs:
    name: folder name
    fps: frames per second
    episodes: list of episodes to include in the gif
    '''
    print('Creating gif for', name)
    images = []
    filenames = os.listdir(f'figures/animations/{name}')
    filenames.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

    for filename in filenames:
        if int(filename.split('_')[0]) in episodes:
            images.append(imageio.imread(f'figures/animations/{name}/' + filename))
    imageio.mimsave(f'figures/animations/{name}.gif', images, duration=1000/fps)

def MakeVideo(name: str, fps: int = 4, freeze: float = 1):
    '''
    Create a video from a series of images in a folder
    name: folder name
    fps: frames per second
    freeze: number of seconds to freeze the image at the beginning and end of the episode
    '''
    print('Creating video for', name)
    images = glob.glob(f'figures/animations/{name}/*.png')
    images.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1].split('.')[0])))
    height, width, _ = cv2.imread(images[0]).shape

    img_array = []
    for i, filename in enumerate(images):
        if filename.split('/')[-1].split('_')[1].split('.')[0] == '0' or filename.split('/')[-1].split('_')[1].split('.')[0] == '99':
            for _ in range(int(freeze * fps)):
                img_array.append(cv2.imread(filename))
                
        img = cv2.imread(filename)
        img_array.append(img)

    out = cv2.VideoWriter(f'figures/animations/{name}.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    # save the video
    for img in img_array:
        out.write(img)


if __name__ == '__main__':
    from EnvironmentWrapper import CustomEnvWrapper
    from stable_baselines3 import SAC

    # Sharpening AO system easy
    experiment_name = "Sharpening_AO_system_easy"
    env = CustomEnvWrapper(name=experiment_name)
    eval_episodes = 10
    eval_steps = 100

    # evaluate no agent
    average_reward = Evaluate(env, eval_episodes, eval_steps)

    # evaluate agent
    model_name = 'SAC-1.7rms-21act-100000buf-2'
    model = SAC.load(f"models/{model_name}", custom_objects={"action_space": env.action_space, "observation_space": env.observation_space})
    average_reward = Evaluate(env, eval_episodes, eval_steps, model, model_name)

    # make gifs
    MakeGif('no_agent', episodes=[5, 6])
    MakeGif(model_name, episodes=[5, 6])

    # make video
    MakeVideo('no_agent', fps=10, freeze=1)
    MakeVideo(model_name, fps=10, freeze=1)


    # Centering AO system
    experiment_name = "Centering_AO_system"
    env = CustomEnvWrapper(name=experiment_name)
    eval_episodes = 10
    eval_steps = 100

    # evaluate no agent
    average_reward = Evaluate(env, eval_episodes, eval_steps, agent_name='no_agent_Centering')

    # evaluate agent
    model_name = 'SAC-3rms-3act-1000buf-3.zip'
    model = SAC.load(f"models/{model_name}", custom_objects={"action_space": env.action_space, "observation_space": env.observation_space})

    # if there is zip in the model name, remove it
    if '.zip' in model_name:
        model_name = model_name.split('.zip')[0]

    average_reward = Evaluate(env, eval_episodes, eval_steps, model, model_name)

    # make gifs
    MakeGif('no_agent_Centering', episodes=[5, 6])
    MakeGif(model_name, episodes=[5, 6])

    # make video
    MakeVideo('no_agent_Centering', fps=10, freeze=1)
    MakeVideo(model_name, fps=10, freeze=1)