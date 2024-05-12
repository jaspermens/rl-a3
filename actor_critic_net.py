from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.distributions.categorical import Categorical

# class ActorCriticNet(nn.Module):
#     def __init__(self, n_inputs, n_actions):
#         super().__init__()
#         self.n_inputs = n_inputs
#         self.n_actions = n_actions

#         n_nodes = 128
#         self.layer1 = nn.Linear(n_inputs, n_nodes)
#         self.act1 = nn.Tanh()
#         self.layer2 = nn.Linear(n_nodes, n_nodes)
#         self.act2 = nn.Tanh()
#         self.layer3 = nn.Linear(n_nodes, n_actions)

#         self.value_output = nn.Linear(n_nodes, 1)
        
#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.Tensor(np.array([x]))
#         x = self.act1(self.layer1(x))
#         x = self.act2(self.layer2(x))
#         return self.layer3(x)
    
#     def get_policy(self, x):
#         return Categorical(logits=self.forward(x)) # TODO: props or logits??
    
#     def get_action(self, x):
#         return self.get_policy(x).sample().item()

#     def get_value(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.Tensor(np.array([x]))

#         x = self.act1(self.layer1(x))
#         x = self.act2(self.layer2(x))
#         return self.value_output(x)
    

class ActorNet(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions

        n_nodes = 64
        self.layer1 = nn.Linear(n_inputs, n_nodes)
        self.act1 = nn.Tanh()
        self.layer2 = nn.Linear(n_nodes, n_nodes)
        self.act2 = nn.Tanh()        
        self.layer3 = nn.Linear(n_nodes, n_nodes)
        self.act3 = nn.Tanh()
        # self.layer4 = nn.Linear(n_nodes, n_nodes)
        # self.act4 = nn.Tanh()
        self.layer5 = nn.Linear(n_nodes, n_actions)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(np.array([x]))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        # x = self.act4(self.layer4(x))
        return self.layer5(x)
    
    def get_policy(self, x):
        return Categorical(logits=self.forward(x)) # TODO: props or logits??
    
    def get_action(self, x):
        return self.get_policy(x).sample().item()


class CriticNet(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_actions = n_actions
        n_nodes = 64
        self.layer1 = nn.Linear(n_inputs, n_nodes)
        self.act1 = nn.Tanh()
        self.layer2 = nn.Linear(n_nodes, n_nodes)
        self.act2 = nn.Tanh()        
        self.layer3 = nn.Linear(n_nodes, n_nodes)
        self.act3 = nn.Tanh()
        # self.layer4 = nn.Linear(n_nodes, n_nodes)
        # self.act4 = nn.Tanh()

        self.value_output = nn.Linear(n_nodes, 1)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.Tensor(np.array([x]))
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        # x = self.act3(self.layer3(x))
        # x = self.act4(self.layer4(x))
    
        return self.value_output(x)
    
    def get_value(self, x):
        return self.forward(x)
    