from dataclasses import dataclass

@dataclass
class ModelParameters:
    envname: str                                # gym environment we'll be training in
    agent_type: str
    lr: float                                   # learning rate
    gamma: float = 1                            
    entropy_reg_factor: float = .1
    early_stopping_return: int | None = None    # critical reward value for early stopping
    backup_depth: int = 100
    eval_interval: int = 2000                      # evaluate every N training steps
    n_eval_episodes: int = 20                    # average eval rewards over N episodes
    num_training_steps: int = 3e5
    do_bootstrap: bool = True
    do_baseline_sub: bool = True

    def __post_init__(self) -> None:
        if not (self.do_bootstrap or self.do_baseline_sub and self.agent_type == "actor_critic"):
            raise UserWarning("Need at least either bootstrapping or baseline subtraction for actor-critic!!") 
