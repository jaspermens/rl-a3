import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

import gymnasium as gym

import torch
from torch import nn
from torch.optim import Adam

from typing import Protocol

from model_parameters import ModelParameters
from actor_critic_net import ActorNet, CriticNet
from reinforce_policy_net import PolicyNet


class PolicyBasedAgent(Protocol):
    params: ModelParameters
    
    def apply_gradient_update(self):
        raise NotImplementedError

    def select_action(self, state):
        raise NotImplementedError

    def take_step(self, state, env: gym.Env):
        action = self.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        return state, action, reward, done

    @torch.no_grad
    def demonstrate(self, n_episodes: int) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""
        env = gym.make(self.params.envname, render_mode="human")
        env.reset(seed=4309)

        for _ in range(n_episodes):
            state, _ = env.reset()  # Uses the newly created environment with render=human
            done = False
            
            while not done:
                state, _, _, done = self.take_step(state, env=env)

            env.reset()

    def make_trail_plot(self, n_episodes: int) -> None:
        env = gym.make(self.params.envname)
        env.reset(seed=4309)

        fig, ax = plt.subplots(1,1, layout='constrained', dpi=150, figsize=[5,5])
        ax.xaxis.set_tick_params(direction='in')
        ax.yaxis.set_tick_params(direction='in')

        for _ in range(n_episodes):
            state, _ = env.reset()
            done = False
            
            step = 0
            positions = []
            rotations = []
            while not done:
                state, _, _, done = self.take_step(state, env=env)

                if step % 10 == 0:
                    positions.append((state[0], state[1]))
                    rotations.append(state[4])

                step += 1

            ax.plot(*np.array(positions).T, marker='.', markersize=4, linewidth=1, alpha=.8, color='black')
            for i, rotation in enumerate(rotations):
                ax.scatter(*positions[i], marker=MarkerStyle('8', 'top', transform=Affine2D().rotate(rotation)), color='red', s=100)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 2)

        ax.set_aspect('equal')

        plt.savefig("trail_plot.png")
        plt.show()


class REINFORCEAgent(PolicyBasedAgent):
    def __init__(self, params: ModelParameters) -> None:
        env = gym.make(params.envname)  
        n_inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()
        
        self.policy_net = PolicyNet(n_actions=n_actions, n_inputs=n_inputs)
        
        self.policy_optimizer = Adam(self.policy_net.parameters(), lr=params.lr)
        
        self.params = params

    def _policy_loss(self, states, actions, backup_targets) -> torch.Tensor:
        policy = self.policy_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        
        loss = - (log_probabilities * backup_targets + self.params.entropy_reg_factor * policy.entropy()).sum()
        return loss
    
    def apply_gradient_update(self, ep_states, ep_actions, ep_rewards):
        backup_targets = np.zeros_like(ep_rewards)
        for t, _ in enumerate(ep_rewards):
            backup_targets[t] = np.sum([self.params.gamma**(k-t) * ep_rewards[k] for k in np.arange(t, len(ep_rewards)-1)])
            # could do gradient update here (i.e., online)
            # or, just sum all of the gradients over the 'batch', which is easier to implement imo
            # so: batch mode (same as our AC implementation)

        # compute and bp loss
        policy_loss = self._policy_loss(
            torch.as_tensor(np.array(ep_states)), 
            torch.as_tensor(ep_actions), 
            torch.as_tensor(backup_targets),
        )
        
        self.policy_optimizer.zero_grad() # remove gradients from previous steps
        policy_loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) # clip gradients
        self.policy_optimizer.step()      # apply gradients
        

    def select_action(self, state):
        return self.policy_net.get_action(state)


class ActorCriticAgent(PolicyBasedAgent):
    def __init__(self, params: ModelParameters) -> None:
        env = gym.make(params.envname)  
        n_inputs = env.observation_space.shape[0]
        n_actions = env.action_space.n
        env.close()
        
        self.actor_net = ActorNet(n_actions=n_actions, n_inputs=n_inputs)
        self.critic_net = CriticNet(n_actions=n_actions, n_inputs=n_inputs)

        self.policy_optimizer = Adam(self.actor_net.parameters(), lr=params.lr)
        self.value_optimizer = Adam(self.critic_net.parameters(), lr=params.lr)
        
        self.params = params

    def _policy_loss(self, states, actions, backup_targets) -> torch.Tensor:
        policy = self.actor_net.get_policy(states)
        log_probabilities = policy.log_prob(actions)
        return - (log_probabilities * backup_targets + self.params.entropy_reg_factor * policy.entropy()).sum()

    def _nstep_backup_targets(self, states, actions, rewards):
        targets = np.zeros_like(actions)
        for t in range(len(states)):
            max_k = np.minimum(t + self.params.backup_depth, len(states)-1) - t
            targets[t] = np.sum([rewards[t+k] * self.params.gamma**k for k in range(max_k)]) \
                + self.params.gamma**self.params.backup_depth * self.critic_net.get_value(states[t + max_k]) 

        return targets
    
    def _mc_backup_targets(self, rewards):
        backup_targets = np.zeros_like(rewards)
        for t, _ in enumerate(rewards):
            backup_targets[t] = np.sum([self.params.gamma**(k-t) * rewards[k] for k in np.arange(t, len(rewards)-1)])

        return backup_targets
    
    def apply_gradient_update(self, ep_states, ep_actions, ep_rewards):

        if self.params.do_bootstrap:
            backup_targets = torch.as_tensor(self._nstep_backup_targets(ep_states, ep_actions, ep_rewards))

        else:
            backup_targets = torch.as_tensor(self._mc_backup_targets(ep_rewards))

        values = self.critic_net.get_value(np.array(ep_states)).squeeze()
        advantages = backup_targets - values.detach()

        # compute and bp loss
        policy_loss = self._policy_loss(
            torch.as_tensor(np.array(ep_states)), 
            torch.as_tensor(ep_actions), 
            advantages if self.params.do_baseline_sub else backup_targets,
        )
        
        self.policy_optimizer.zero_grad() # remove gradients from previous steps
        policy_loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.actor_net.parameters(), 100) # clip gradients
        self.policy_optimizer.step()      # apply gradients
        
        value_loss = (backup_targets - values).square().sum()
        
        self.value_optimizer.zero_grad() # remove gradients from previous steps
        loss = torch.as_tensor(value_loss)

        loss.backward()            # compute gradients
        nn.utils.clip_grad_value_(self.critic_net.parameters(), 100) # clip gradients
        self.value_optimizer.step()      # apply gradients

    def select_action(self, state):
        return self.actor_net.get_action(state)
    

agents: dict[str, PolicyBasedAgent] = {'actor_critic': ActorCriticAgent, 'REINFORCE': REINFORCEAgent}