import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandbCustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCustomCallback, self).__init__(verbose=verbose)

    # get the reward every step
    def _on_step(self) -> bool:
        # Log scalar values (here a random variable)
        wandb.log({"reward": self.locals["rewards"]})
        return True
    
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self):
        # Append the reward to the list after each step
        self.rewards.append(self.locals["rewards"])
        return True
    def reset(self):
        self.rewards = []
