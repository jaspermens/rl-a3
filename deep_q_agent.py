import torch
import numpy as np
import gymnasium as gym 

from data_handling import ReplayBuffer
from dqn import DeepQModel
from policies import Policy


class DeepQAgent:
    def __init__(self, env: gym.Env, 
                 policy: Policy, 
                 exploration_parameter: float,
                 buffer_capacity: int,
                 ):
        self.env = env
        self.reset()

        self.policy = policy
        self.exploration_parameter = exploration_parameter
        self.buffer_capacity = buffer_capacity

        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        self.burn_in()
        
    def reset(self):
        self.state, _ = self.env.reset()

    def select_action(self, model: DeepQModel):
        # basically self.model.evaluate(state)
        state = torch.tensor(np.array([self.state]))

        # get q values:
        q_values = model.forward(state)

        # greedy best action:
        action = self.policy(q_values, self.exploration_parameter)

        return action

    @torch.no_grad      # disable gradient calculation here. I think it saves memory
    def take_step(self, action):
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if done:
            new_state = None

        self.buffer.append(state=self.state, action=action, reward=reward, new_state=new_state, done=done)
        
        self.state = new_state           
        
        if done:
            self.reset()

        return reward, done
    
    @torch.no_grad
    def act_on_model(self, model: DeepQModel):
        action = self.select_action(model = model)
        reward, done = self.take_step(action=action)     
    
        return reward, done
    
    @torch.no_grad    
    def burn_in(self):
        # randomly populate the buffer
        mock_q_values = np.ones(self.env.action_space.n)
        for _ in range(self.buffer_capacity):
            action = Policy.RANDOM(mock_q_values)
            self.take_step(action)
        