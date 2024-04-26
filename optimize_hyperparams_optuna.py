import optuna
import numpy as np
from babymode_actorcritic import LunarLanderAC

def objective(trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", .9, 1)
    entropy_reg_factor = trial.suggest_float("eta", 1e-5, 1, log=True)
    backup_depth = trial.suggest_int("backup_depth", 10, 200, step=5)
    
    model_params = {
        'lr': lr,
        'gamma': gamma,
        'early_stopping_return': None,
        'entropy_reg_factor': entropy_reg_factor,
        'backup_depth': backup_depth,
        'envname': "LunarLander-v2",
        'num_training_steps': 1e5,
    }

    finalscores = []
    num_repetitions = 3
    for _ in range(num_repetitions):
        actorcritic = LunarLanderAC(**model_params)
        actorcritic.train_model()
    
        finalscores.append(actorcritic.eval_returns[-1])

    return np.mean(finalscores)

def do_study():
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.GPSampler)
    study.optimize(objective, n_trials=10)
    
    print("best params:", study.best_params)
    print("best value:", study.best_value)


if __name__ == '__main__':
    do_study()