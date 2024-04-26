import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from reinforce_policy_net import PolicyNet

# TODO: early stopping fix
# TODO: test/remove batch thing -> do we need it? seems like a weird thing to try first yknow

class LunarLanderREINFORCE:
    """Main class implementing the REINFORCE algorithm on box2d gym environments"""
    def __init__(self, 
                 envname: str,                      # gym environment we'll be training in
                 lr: float,                         # learning rate
                 batch_size: int,                   # (only used with experience replay)
                 gamma: float = 1,                  
                 entropy_reg_factor: float = .1,
                 early_stopping_return: int | None = None, # critical reward value for early stopping
                 backup_depth: int = 10,
                 eval_interval: int = 2000,     # evaluate every N training steps
                 n_eval_episodes: int = 5,        # average eval rewards over N episodes
                 ):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.envname = envname
        self.env = gym.make(envname)  
        self.eval_env = gym.make(envname)
        self.entropy_reg_factor = entropy_reg_factor
        self.n_steps = backup_depth
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes

        if early_stopping_return is None:   # if not specified, then take from env
            self.early_stopping_return = self.env.spec.reward_threshold
        else:
            self.early_stopping_return = early_stopping_return

        # set various counters, lists, etc
        self.reset_counters()
        
        # init the model and agent
        n_inputs = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.policy_net = PolicyNet(n_inputs=n_inputs, n_actions=n_actions)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        
    def reset_counters(self):
        self.episode_returns = [] 
        self.episode_times = []
        self.eval_returns = []
        self.eval_times = []
        self.total_time = 0


    def train_batch_reinforce(self, for_eval: bool = False, freeze_gradients: bool = False):
        """Fills one batch with experiences and updates the network"""
        
        env = self.eval_env if for_eval else self.env
        state, _ = self.env.reset()

        done = False
        batch_states = []
        batch_weights = []
        batch_actions = []
        batch_returns = []
        episode_return = 0
        episode_length = 0
        
        while True:
            batch_states.append(state)

            action = self.policy_net.get_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # append experience
            episode_return += reward
            batch_actions.append(action)
            episode_length += 1
            self.total_time += 1

            if not for_eval:
                if self.total_time % self.eval_interval == 0:
                    self.evaluate_model()

            if done:    
                # compute episode rewards etc
                batch_returns.append(episode_return)
                # use gamma here???
                if not freeze_gradients:
                    self.episode_returns.append(episode_return)
                    self.episode_times.append(self.total_time)

                batch_weights += [episode_return * self.gamma**i for i in range(episode_length)[::-1]]
                
                state, _ = self.env.reset()
                done = False
                episode_return = 0
                episode_length = 0

                if for_eval:
                    return np.mean(batch_returns)
                elif len(batch_states) > self.batch_size:
                    break

        # compute and bp loss

        if freeze_gradients:
            return np.mean(batch_returns)
        
        self.optimizer.zero_grad() # remove gradients from previous steps
        loss = self.loss_function(torch.as_tensor(np.array(batch_states)), 
                                  torch.as_tensor(batch_actions), 
                                  torch.as_tensor(batch_weights))
        loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # clip gradients
        self.optimizer.step()      # apply gradients

        return np.mean(batch_returns)
    

    def evaluate_model(self, store_output: bool = True):
        episode_scores = []
        for _ in range(self.n_eval_episodes):
            episode_score = self.train_batch_reinforce(for_eval=True)
            episode_scores.append(episode_score)

        mean_return = np.mean(episode_scores)
        if store_output:
            self.eval_returns.append(mean_return)
            self.eval_times.append(self.total_time)
        return mean_return


    def loss_function(self, states, actions, weights):
        policy = self.policy_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * weights + self.entropy_reg_factor * policy.entropy()).mean()


    def train_model(self, num_episodes: int):
        for _ in tqdm(range(num_episodes), total=num_episodes):
            batch_return = self.train_batch_reinforce()
            if batch_return < self.early_stopping_return:
                continue

            if self.do_early_stopping():
                print("STOPPING EARLY LOL")
                break

    def do_early_stopping(self):
        eval_score = self.evaluate_model(store_output=False)
        return eval_score >= self.early_stopping_return

    def render_run(self, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""
        env = gym.make(self.envname, render_mode="human")
        env.reset(seed=4309)
        for _ in range(n_episodes_to_plot):
            state, _ = env.reset()  # Uses the newly created environment with render=human
            done = False
            
            while not done:
                state = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    action = self.policy_net.get_action(state)
                state, _, terminated, truncated, _ = env.step(action=action)

                done = terminated or truncated

            env.reset()
        
    def plot_learning(self):
        fig, ax = plt.subplots()
        ax.grid(True, alpha=.5)
        ax.plot(self.eval_times, self.eval_returns)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Training Episode Return")
        plt.show()


def train_reinforce_model(): 
    model_params = {
            'lr': 0.001,
            'batch_size': 1024,
            'gamma': .99,
            'early_stopping_return': None,
            'entropy_reg_factor': 0.1,
            'backup_depth': 500,
            'envname': "LunarLander-v2"
    }

    env = gym.make("LunarLander-v2")
    reinforcer = LunarLanderREINFORCE(**model_params)

    try:
        reinforcer.train_model(num_episodes=100)
    except KeyboardInterrupt:
        pass
    reinforcer.render_run(n_episodes_to_plot=10)
    reinforcer.plot_learning()

if __name__ == "__main__":
    train_reinforce_model()