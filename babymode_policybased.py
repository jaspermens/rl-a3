import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import torch

from model_parameters import ModelParameters
from agents import ActorCriticAgent 
from agents import agents


class PolicyTrainer:
    """Main class handling the training of an actor-critic model in box2d gym environments"""
    def __init__(self, params: ModelParameters):
        self.params = params
        self.env = gym.make(params.envname)  
        self.eval_env = gym.make(params.envname)

        self.agent: ActorCriticAgent = agents[params.agent_type](params)

        if params.early_stopping_return is None:   # if not specified, then take from env
            self.early_stopping_return = self.env.spec.reward_threshold
        else:
            self.early_stopping_return = params.early_stopping_return

        # set various counters, lists, etc
        self.reset_counters()
        
    def reset_counters(self) -> None:
        self.episode_returns = [] 
        self.episode_times = []
        self.eval_returns = []
        self.eval_times = []
        self.final_reward = 0
        self.total_time = 0

    def train_model(self) -> None:
        # for _ in tqdm(range(num_episodes), total=num_episodes):
        pbar = tqdm(total=self.params.num_training_steps)
        curtime = self.total_time                               # needed for progress bar
        while self.total_time < self.params.num_training_steps:


            pbar.update(self.total_time - curtime)              # advance the pbar
            curtime = self.total_time                           #

            ep_states, ep_actions, ep_rewards, stop_early = self.perform_trace()
            if stop_early:
                print("STOPPING EARLY LOL")
                break

            self.agent.apply_gradient_update(ep_states=ep_states, ep_actions=ep_actions, ep_rewards=ep_rewards)

        pbar.close()

    def perform_trace(self) -> tuple[list, list, list, bool]:
        """Performs one epoch of actor-critic training, and return everything needed for gradient updates etc."""
        env = self.env
        
        state, _ = env.reset()

        trace_states = []
        trace_rewards = []
        trace_actions = []
        episode_return = 0
        episode_length = 0

        done = False
        while not done:
            trace_states.append(state)

            action = self.agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # append experience
            episode_return += reward
            trace_actions.append(action)
            trace_rewards.append(reward)
            episode_length += 1
            if (self.total_time + episode_length) % self.params.eval_interval == 0:
                can_stop_early = self.evaluate_model(time=self.total_time + episode_length)
                if can_stop_early:
                    return None, None, None, True
                
        self.total_time += episode_length
        self.episode_returns.append(episode_return)
        self.episode_times.append(self.total_time)

        return trace_states, trace_actions, trace_rewards, False

        
    @torch.no_grad
    def evaluate_model(self, time: int | None = None) -> float:
        episode_scores = []
        for _ in range(self.params.n_eval_episodes):
            episode_score = self.perform_eval_episode()
            episode_scores.append(episode_score)

        mean_return = np.mean(episode_scores)

        self.eval_returns.append(mean_return)
        self.eval_times.append(time)

        return mean_return >= self.early_stopping_return

    @torch.no_grad
    def perform_eval_episode(self) -> float:
        state, _ = self.eval_env.reset()

        episode_return = 0

        done = False
        while not done:
            state, _, reward, done = self.agent.take_step(state=state, env=self.eval_env)

            episode_return += reward

        return episode_return

    def render_run(self, n_episodes_to_plot: int = 10) -> None:
        """Runs a single evaluation episode while rendering the environment for visualization."""
        self.agent.demonstrate(n_episodes_to_plot)
        
    def plot_learning(self):
        fig, ax = plt.subplots()
        ax.grid(True, alpha=.5)
        ax.plot(self.eval_times, self.eval_returns)
        ax.plot(self.episode_times, self.episode_returns, alpha=.3)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Training Episode Return")
        plt.show()


def train_reinforce_model(): 
    model_params = ModelParameters(
            lr = 1e-3,
            gamma = .97,
            early_stopping_return = None,
            entropy_reg_factor = 0.01,
            backup_depth = 200,
            envname = "CartPole-v1",
            num_training_steps = 1e5, 
            do_bootstrap = False, 
            do_baseline_sub = True, 
            agent_type = 'actor_critic', 
    )
    
    trainer = PolicyTrainer(model_params)

    try:
        trainer.train_model()
    except KeyboardInterrupt:
        pass
    
    trainer.plot_learning()
    trainer.render_run(n_episodes_to_plot=10)


if __name__ == "__main__":
    train_reinforce_model()