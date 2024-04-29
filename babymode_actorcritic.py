import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from actor_critic_net import ActorNet, CriticNet
from model_parameters import ModelParameters

class LunarLanderAC:
    """Main class handling the training of an actor-critic model in box2d gym environments"""
    def __init__(self, params: ModelParameters):
        
        self.params = params
        self.env = gym.make(params.envname)  
        self.eval_env = gym.make(params.envname)

        if params.early_stopping_return is None:   # if not specified, then take from env
            self.early_stopping_return = self.env.spec.reward_threshold
        else:
            self.early_stopping_return = params.early_stopping_return

        # set various counters, lists, etc
        self.reset_counters()
        
        # init the model and agent
        n_inputs = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        # self.ac_net = ActorCriticNet(n_inputs=n_inputs, n_actions=n_actions)
        self.actor_net = ActorNet(n_actions=n_actions, n_inputs=n_inputs)
        self.critic_net = CriticNet(n_actions=n_actions, n_inputs=n_inputs)
        # TODO: value network - extra network or just extra output head?

        # TODO: separate optimizers?
        self.policy_optimizer = Adam(self.actor_net.parameters(), lr=params.lr)
        self.value_optimizer = Adam(self.critic_net.parameters(), lr=params.lr)
        
    def reset_counters(self):
        self.episode_returns = [] 
        self.episode_times = []
        self.eval_returns = []
        self.eval_times = []
        self.final_reward = 0
        self.total_time = 0

    def train_episode_actorcritic(self, for_eval: bool = False):
        """Performs one epoch of actor-critic training"""

        def nstep_backup_targets(states, actions, rewards):
            targets = np.zeros_like(actions)
            for t in range(len(states)):
                max_k = np.minimum(t + self.params.backup_depth, len(states)-1) - t
                targets[t] = np.sum([rewards[t+k]*self.params.gamma**k for k in range(max_k)]) + self.params.gamma**self.params.backup_depth * self.critic_net.get_value(states[max_k]) 

            return targets
        
        env = self.eval_env if for_eval else self.env
        
        state, _ = env.reset()

        done = False
        trace_states = []
        trace_rewards = []
        trace_actions = []
        episode_return = 0
        episode_length = 0

        while not done:
            trace_states.append(state)

            action = self.actor_net.get_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # append experience
            episode_return += reward
            trace_actions.append(action)
            trace_rewards.append(reward)
            episode_length += 1
            if (self.total_time + episode_length) % self.params.eval_interval == 0 and not for_eval:
                self.evaluate_model(time=self.total_time + episode_length, store_output=True)

        if for_eval:
            return episode_return
        
        self.total_time += episode_length
        # use gamma here???
        self.episode_returns.append(episode_return)
        self.episode_times.append(self.total_time)

        backup_targets = nstep_backup_targets(trace_states, trace_actions, trace_rewards)
        # value_loss = torch.as_tensor(np.mean((backup_targets - self.critic_net.get_value(np.array(trace_states)).numpy())**2))
        but = torch.as_tensor(backup_targets)
        values = self.critic_net.get_value(np.array(trace_states)).squeeze()
        trace_advantages = but - values

        # compute and bp loss
        policy_loss = self.policy_loss_baseline_subtracted(
            torch.as_tensor(np.array(trace_states)), 
            torch.as_tensor(trace_actions), 
            trace_advantages.detach(),
        )
        
        self.policy_optimizer.zero_grad() # remove gradients from previous steps
        policy_loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.actor_net.parameters(), 100) # clip gradients
        self.policy_optimizer.step()      # apply gradients
        
        value_loss = trace_advantages.square().sum()
        
        self.value_optimizer.zero_grad() # remove gradients from previous steps
        loss = torch.as_tensor(value_loss)

        loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.critic_net.parameters(), 100) # clip gradients
        self.value_optimizer.step()      # apply gradients

        return episode_return
    
    def evaluate_model(self, time: int | None = None, store_output: bool = True):
        episode_scores = []
        for _ in range(self.params.n_eval_episodes):
            episode_score = self.train_episode_actorcritic(for_eval=True)
            episode_scores.append(episode_score)

        mean_return = np.mean(episode_scores)
        if store_output:
            self.eval_returns.append(mean_return)
            self.eval_times.append(time)

        return mean_return

    def policy_loss_function(self, states, actions, backup_targets) -> torch.Tensor:
        policy = self.actor_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * backup_targets + self.params.entropy_reg_factor * policy.entropy()).sum()

    def policy_loss_baseline_subtracted(self, states, actions, advantages) -> torch.Tensor:
        policy = self.actor_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * advantages + self.params.entropy_reg_factor * policy.entropy()).sum()

    def train_model(self) -> None:
        # for _ in tqdm(range(num_episodes), total=num_episodes):
        pbar = tqdm(total=self.params.num_training_steps)
        curtime = 0
        while self.total_time < self.params.num_training_steps:
            pbar.update(self.total_time - curtime)
            curtime = self.total_time
            batch_return = self.train_episode_actorcritic()
            if batch_return < self.early_stopping_return:
                continue
            if self.do_early_stopping():
                print("STOPPING EARLY LOL")
                break
        else:
            self.final_reward = self.evaluate_model(store_output=False)
        
        pbar.close()

    def do_early_stopping(self):
        eval_score = self.evaluate_model(store_output=False)
        self.final_reward = eval_score
        return eval_score >= self.early_stopping_return

    def render_run(self, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""
        env = gym.make(self.params.envname, render_mode="human")
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

            env.reset()
        
    def plot_learning(self):
        fig, ax = plt.subplots()
        ax.grid(True, alpha=.5)
        ax.plot(self.eval_times, self.eval_returns)
        ax.plot(self.episode_times, self.episode_returns, alpha=.3)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Training Episode Return")
        plt.show()


def train_reinforce_model(): 
    model_params = ModelParameters(**{
            'lr': 5e-4,
            'gamma': .99,
            'early_stopping_return': None,
            'entropy_reg_factor': 0.1,
            'backup_depth': 100,
            'envname': "CartPole-v1",
            'num_training_steps': 100_000, 
    })
    
    reinforcer = LunarLanderAC(model_params)

    try:
        reinforcer.train_model()
    except KeyboardInterrupt:
        pass
    
    reinforcer.plot_learning()
    reinforcer.render_run(n_episodes_to_plot=10)


if __name__ == "__main__":
    train_reinforce_model()