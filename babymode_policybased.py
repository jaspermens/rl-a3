import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from policies import Policy
from dqn import DeepQModel

# TODO: Policy
# TODO: early stopping fix


class LunarLanderREINFORCE:
    """Main class handling the training of a DQN in a Gym environment"""
    def __init__(self, 
                 env: gym.Env,                      # gym environment we'll be training in
                 lr: float,                         # learning rate
                 batch_size: int,                   # (only used with experience replay)
                 gamma: float = 1,                  
                 entropy_reg_factor: float = .1,
                 early_stopping_return: int | None = None, # critical reward value for early stopping
                 ):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.env = env  
        self.entropy_reg_factor = entropy_reg_factor

        if early_stopping_return is None:   # if not specified, then take from env
            self.early_stopping_return = env.spec.reward_threshold
        else:
            self.early_stopping_return = early_stopping_return

        # set various counters, lists, etc
        self.reset_counters()
        
        # init the model and agent
        n_inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.policy_net = DeepQModel(n_inputs=n_inputs, n_actions=n_actions)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)
        
    def reset_counters(self):
        self.episode_returns = [] 
        self.episode_times = []
        self.total_time = 0


    def train_batch_reinforce(self, freeze_gradients: bool = False):
        """Fills one batch with experiences and updates the network"""
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

                if len(batch_states) > self.batch_size:
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
    
    def loss_function(self, states, actions, weights):
        policy = self.policy_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * weights + self.entropy_reg_factor * policy.entropy()).mean()


    def train_model(self, num_episodes: int):
        for _ in tqdm(range(num_episodes), total=num_episodes):
            batch_return = self.train_batch_reinforce()
            if batch_return >= self.early_stopping_return:
                stop_early = self.do_early_stopping()
                if stop_early:
                    print("STOPPING EARLY LOL")
                    break

    def do_early_stopping(self):
        eval_returns = []
        for _ in range(5):
            eval_return = self.train_batch_reinforce(freeze_gradients=True)
            eval_returns.append(eval_return)

        if np.mean(eval_returns) >= self.early_stopping_return:
            return True

        return False

    def dqn_render_run(self, env: gym.Env, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""

        env.reset(seed=4309)
        for _ in range(n_episodes_to_plot):
            state, _ = env.reset()  # Uses the newly created environment with render=human
            done = False
            
            while not done:
                state = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.policy_net.forward(state)
                action = Policy.GREEDY(q_values)
                state, _, terminated, truncated, _ = env.step(action=action)

                done = terminated or truncated

            self.env.reset()
        
    def plot_learning(self):
        fig, ax = plt.subplots()
        ax.grid(True, alpha=.5)
        ax.plot(self.episode_times, self.episode_returns)
        plt.show()


def train_reinforce_model(): 
    model_params = {
            'lr': 0.001,  
            'batch_size': 1024,
            'gamma': .999,
            'early_stopping_return': 100,
            'entropy_reg_factor': 0.01,
    }

    env = gym.make("LunarLander-v2")
    reinforcer = LunarLanderREINFORCE(env=env, **model_params)

    reinforcer.train_model(num_episodes=500)
    watch_env = gym.make("LunarLander-v2", render_mode='human')
    reinforcer.plot_learning()
    reinforcer.dqn_render_run(env=watch_env)

if __name__ == "__main__":
    train_reinforce_model()