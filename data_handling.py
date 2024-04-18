import numpy as np
import random
import torch

from collections import deque, namedtuple


Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state"],
)


class ReplayBuffer:
    """Cyclical buffer for holding an agent's recent experiences"""
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, done, new_state) -> None:
        def to_tensor(x):   
            if x is None:   
                return x        # because sometimes new_state is None
            return torch.Tensor(np.array([x]))
        
        exp = Experience(
            to_tensor(state),
            to_tensor(int(action)),
            to_tensor(reward),
            to_tensor(done),
            to_tensor(new_state),
        )

        self.buffer.append(exp)

    def sample(self, batch_size: int):
        """Sample a batch from the buffer"""
        return random.sample(self.buffer, batch_size)