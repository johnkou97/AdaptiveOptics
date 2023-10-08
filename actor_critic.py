import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
from helper import linear_anneal, smooth
from gym_ao.gym_ao.gym_sharpening import Sharpening_AO_system
from gym_ao.gym_ao.gym_darkhole import Darkhole_AO_system


parser = argparse.ArgumentParser()
parser.add_argument("--bootstrap", default=False, action="store_true", help="Use bootstrap")
parser.add_argument("--baseline", default=False, action="store_true", help="Use baseline subtraction")
args = parser.parse_args()


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state.unsqueeze(0).view(-1, self.state_dim))


# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        state = state.float()
        return self.net(state.unsqueeze(0).view(-1, self.state_dim))


# Define the TraceElement class
class TraceElement:
    def __init__(self, state, action, reward, next_state, done, log_prob, entropy):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob
        self.entropy = entropy


class HyperparamsActorCritic:
    def __init__(self, gamma: float = 0.99, lr_actor: float = 0.005, lr_critic: float = 0.05, hidden_units: int = 128,
                 optimizer: str = "Adam", entropy_reg: float = 0.01, batch_size: int = 4, n_steps: int = 1,
                 bootstrap: bool = True, baseline: bool = True, anneal: bool = False, min_entro_reg: float = 0.001,
                 perc: float = 0.6, trace_length: int = 1000):
        self.bootstrap = bootstrap
        self.baseline = baseline
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.entropy_reg = entropy_reg
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.anneal = anneal
        self.min_entro_reg = min_entro_reg
        self.perc = perc
        self.trace_length = trace_length

    def dict(self):
        return {"gamma": self.gamma, "lr_actor": self.lr_actor, "lr_critic": self.lr_critic,
                "hidden_units": self.hidden_units, "optimizer": self.optimizer, "entropy_reg": self.entropy_reg,
                "batch_size": self.batch_size, "n_steps": self.n_steps, "bootstrap": self.bootstrap,
                "baseline": self.baseline, "anneal": self.anneal, "min_entro_reg": self.min_entro_reg,
                "perc": self.perc, "trace_length": self.trace_length}


# Define the Actor-Critic algorithm with bootstrapping and baseline subtraction
class ActorCritic:
    def __init__(self, env, hyperparams: HyperparamsActorCritic, experiment_name: str = "actor_critic"):
        self.env = env
        state_dim = np.prod(self.env.observation_space.shape)
        action_dim = self.env.action_space.shape[0]
        self.experiment_name = experiment_name
        self.hyperparams = hyperparams
        self.actor = Actor(state_dim, action_dim, hyperparams.hidden_units)
        self.critic = Critic(state_dim, hyperparams.hidden_units)
        self.optimizer_actor = getattr(optim, hyperparams.optimizer)(self.actor.parameters(), lr=hyperparams.lr_actor,
                                                                     maximize=True)  # Maximize: gradient ascent
        self.optimizer_critic = getattr(optim, hyperparams.optimizer)(self.critic.parameters(), lr=hyperparams.lr_critic)
        self.gamma = hyperparams.gamma
        self.entropy_reg = hyperparams.entropy_reg
        self.n_step = hyperparams.n_steps
        self.bootstrap = hyperparams.bootstrap
        self.baseline = hyperparams.baseline
        self.trace_length = hyperparams.trace_length

    @staticmethod   
    def compute_entropy(m) -> torch.Tensor:
        """Computes the entropy of the policy model"""
        return - torch.sum(m.probs * m.logits)

    def get_action(self, state):
        state = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0)
        action_mean = self.actor(state)
        action_std = torch.ones_like(action_mean) * 0.1  # You can adjust the standard deviation as needed
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        # Clip the action to ensure it falls within the desired range
        action = torch.clamp(action, -0.3, 0.3)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.squeeze().numpy(), log_prob, entropy




    def generate_trace(self) -> (list, float):
        """Generates a trace of the environment using the current policy model"""
        s = self.env.reset()
        s = torch.tensor(s, dtype=torch.float32)
        trace = []
        reward_trace = 0
        done = False
        for _ in range(self.trace_length):
            a, log_probs, entropy = self.get_action(s)
            next_s, r, done, _, _ = self.env.step(a)
            # print(f'Reward: {r}')
            reward_trace += r
            next_s = torch.tensor(next_s)
            trace.append(TraceElement(s, a, r, next_s, done, log_probs, entropy))
            s = next_s
        # print(f'Episode reward: {reward_trace}')
        return trace, reward_trace

    def bootstrapping(self, trace):
        """Bootstraps the trace with the critic"""
        end = self.n_step
        if len(trace) < self.n_step:
            end = len(trace)
        sum_value = 0
        for k in range(end):
            sum_value += (self.gamma ** k) * trace[k].reward
        sum_value += (self.gamma ** end) * self.critic(trace[end - 1].next_state) * (1 - trace[end - 1].done)
        return sum_value.squeeze(0)

    def baseline_subtraction(self, trace, q_n):
        return q_n - self.critic(trace.state)

    def learn(self, trace):
        """Learns from a trace of the environment"""
        q_n = []
        advantage = []
        for t in range(len(trace)):
            # Bootstrapping
            if self.bootstrap:
                q_n.append(self.bootstrapping(trace[t:]))
            else:
                temp_qn = self.critic(trace[t].state) - self.critic(trace[t].state)  # Keep track of the gradient
                q_n.append(temp_qn)  # Works for Sharpening env
                # q_n.append(temp_qn.detach().numpy()) # Works for both envs
                for k in range(len(trace) - t):
                    q_n[-1] += (self.gamma ** k) * trace[t+k].reward

            # Baseline subtraction
            if self.baseline:
                advantage.append(self.baseline_subtraction(trace[t], q_n[-1]))
            else:
                advantage.append(q_n[-1])
        return torch.stack(q_n), torch.stack(advantage).squeeze(1) # Works for Sharpening env
        # return torch.stack([torch.tensor(q, dtype=torch.float32) for q in q_n]), torch.stack([torch.tensor(a, dtype=torch.float32) for a in advantage]).squeeze(1) # Works for both envs

    def train(self, n_epochs=1000):
        # Train the Actor-Critic algorithm
        train_rewards = []
        for epoch in range(n_epochs):
            loss_actor = 0
            loss_critic = 0
            mean_reward = 0
            for episode in range(self.hyperparams.batch_size):
                # Generate a trace of the environment
                trace, reward_trace = self.generate_trace()
                mean_reward += reward_trace / self.hyperparams.batch_size
                train_rewards.append(reward_trace/ self.hyperparams.trace_length)

                # Learn from the trace
                q_n, advantage = self.learn(trace)
                entropies = torch.stack([t.entropy.unsqueeze(0) for t in trace])
                if self.baseline:
                    episode_loss_critic = advantage.pow(2).sum()
                else:
                    episode_loss_critic = (q_n - self.critic(torch.stack([t.state for t in trace]))).pow(2).sum()
                loss_critic += episode_loss_critic
                log_probs = torch.stack([t.log_prob for t in trace])

                if self.baseline:
                    episode_loss_actor = (advantage.detach() * log_probs + self.entropy_reg * entropies).sum()
                else:
                    episode_loss_actor = (q_n.detach() * log_probs + self.entropy_reg * entropies).sum()
                loss_actor += episode_loss_actor

            loss_actor /= self.hyperparams.batch_size
            loss_critic /= self.hyperparams.batch_size

            # train_rewards.append(mean_reward)
            if epoch % 20 == 0:
                print(f'Episode: {epoch}, average reward last 20 epochs: {np.mean(train_rewards[-20:])}')

            # Update the actor and critic networks
            self.critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

            # Compute the actor loss and update the actor network
            self.actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()
            
            if self.hyperparams.anneal:
                self.entropy_reg = linear_anneal(epoch, n_epochs, self.hyperparams.entropy_reg,
                                                 self.hyperparams.min_entro_reg, self.hyperparams.perc)

        return train_rewards


if __name__ == '__main__':
    # Define the environment
    env = Sharpening_AO_system()
    # env = Darkhole_AO_system()

    env_name = str(env).split('.')[2].split('_')[1]
    print(env_name)

    # Define the Actor-Critic algorithm and train it
    hyperparams_ac = HyperparamsActorCritic(entropy_reg=0.01, lr_actor=0.005, lr_critic=0.05, n_steps=1, batch_size=4, trace_length=10,
                                            bootstrap=args.bootstrap, baseline=args.baseline)
    ac = ActorCritic(env=env, hyperparams=hyperparams_ac, experiment_name='actor-critic')
    rewards = ac.train(n_epochs=1000)
    
    torch.save(ac.actor, env_name + '_actor_critic.pt')  # Save the actor network

    np.save('rewards_' + env_name + '_actor_critic.npy', rewards)  # Save the rewards

    # Plot the rewards
    plt.figure()
    plt.plot(smooth(rewards, 51))
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.savefig(env_name + '_actor_critic.png', dpi=600)
    plt.close()