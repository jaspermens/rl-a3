import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from actor_critic_net import ActorCriticNet, ActorNet, CriticNet

class LunarLanderAC:
    """Main class handling the training of an actor-critic model in box2d gym environments"""
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
        self.n_steps = 15

        if early_stopping_return is None:   # if not specified, then take from env
            self.early_stopping_return = env.spec.reward_threshold
        else:
            self.early_stopping_return = early_stopping_return

        # set various counters, lists, etc
        self.reset_counters()
        
        # init the model and agent
        n_inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # self.ac_net = ActorCriticNet(n_inputs=n_inputs, n_actions=n_actions)
        self.actor_net = ActorNet(n_actions=n_actions, n_inputs=n_inputs)
        self.critic_net = CriticNet(n_actions=n_actions, n_inputs=n_inputs)
        # TODO: value network - extra network or just extra output head?

        # TODO: separate optimizers?
        self.policy_optimizer = Adam(self.actor_net.parameters(), lr=lr)
        self.value_optimizer = Adam(self.critic_net.parameters(), lr=lr)
        
    def reset_counters(self):
        self.episode_returns = [] 
        self.episode_times = []
        self.total_time = 0

    def train_episode_actorcritic(self, freeze_gradients: bool = False):
        """Performs one epoch of actor-critic training"""

        def nstep_backup_targets(states, actions, rewards):
            targets = np.zeros_like(actions)
            for t in range(len(states)):
                max_k = np.minimum(t + self.n_steps, len(states)-1) - t
                targets[t] = np.sum([rewards[t+k]*self.gamma**k for k in range(max_k)]) + self.gamma**self.n_steps * self.critic_net.get_value(states[max_k]) 

            return targets
        
        state, _ = self.env.reset()

        done = False
        trace_states = []
        trace_rewards = []
        trace_actions = []
        episode_return = 0
        episode_length = 0

        while not done:
            trace_states.append(state)

            action = self.actor_net.get_action(state)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # append experience
            episode_return += reward
            trace_actions.append(action)
            trace_rewards.append(reward)
            episode_length += 1
            self.total_time += 1

        # use gamma here???
        if not freeze_gradients:
            self.episode_returns.append(episode_return)
            self.episode_times.append(self.total_time)

        # trace_weights = [episode_return * self.gamma**i for i in range(episode_length)[::-1]]

        if freeze_gradients:
            return episode_return
        
        backup_targets = nstep_backup_targets(trace_states, trace_actions, trace_rewards)
        # value_loss = torch.as_tensor(np.mean((backup_targets - self.critic_net.get_value(np.array(trace_states)).numpy())**2))
        value_loss = (torch.as_tensor(backup_targets) - self.critic_net.get_value(torch.as_tensor(np.array(trace_states)))).square().sum()
        # compute and bp loss
        self.policy_optimizer.zero_grad() # remove gradients from previous steps
        policy_loss = self.policy_loss_function(torch.as_tensor(np.array(trace_states)), 
                                  torch.as_tensor(trace_actions), 
                                  torch.as_tensor(backup_targets))
        policy_loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.actor_net.parameters(), 100) # clip gradients
        self.policy_optimizer.step()      # apply gradients
        
        
        self.value_optimizer.zero_grad() # remove gradients from previous steps
        loss = torch.as_tensor(value_loss)

        loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.critic_net.parameters(), 100) # clip gradients
        self.value_optimizer.step()      # apply gradients

        return episode_return
    
    def policy_loss_function(self, states, actions, backup_targets) -> torch.Tensor:
        policy = self.actor_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * backup_targets + self.entropy_reg_factor * policy.entropy()).mean()


    def train_model(self, num_episodes: int):
        for _ in tqdm(range(num_episodes), total=num_episodes):
            batch_return = self.train_episode_actorcritic()
            if batch_return < self.early_stopping_return:
                continue
            if self.do_early_stopping():
                print("STOPPING EARLY LOL")
                break

    def do_early_stopping(self):
        eval_returns = []
        for _ in range(5):
            eval_return = self.train_episode_actorcritic(freeze_gradients=True)
            eval_returns.append(eval_return)

        if np.mean(eval_returns) >= self.early_stopping_return:
            return True

        return False

    def render_run(self, env: gym.Env, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""

        env.reset(seed=4309)
        for _ in range(n_episodes_to_plot):
            state, _ = env.reset()  # Uses the newly created environment with render=human
            done = False
            
            while not done:
                state = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    action = self.actor_net.get_action(state)
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
    reinforcer = LunarLanderAC(env=env, **model_params)

    reinforcer.train_model(num_episodes=300)
    watch_env = gym.make("LunarLander-v2", render_mode='human')
    reinforcer.plot_learning()
    reinforcer.render_run(env=watch_env)

if __name__ == "__main__":
    train_reinforce_model()