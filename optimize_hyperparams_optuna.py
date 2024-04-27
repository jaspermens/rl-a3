import optuna
import numpy as np
from babymode_actorcritic import LunarLanderAC
from babymode_reinforce import LunarLanderREINFORCE
import pickle as pk

from optuna.visualization.matplotlib import plot_contour
import matplotlib.pyplot as plt


def objective(trial, modeltype: type, num_training_steps: int, num_repeats: int) -> float:
    assert modeltype in [LunarLanderAC, LunarLanderREINFORCE]

    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", .9, 1)
    entropy_reg_factor = trial.suggest_float("eta", 1e-5, 1e-2, log=True)
    backup_depth = trial.suggest_int("backup_depth", 10, 200, step=5)
    
    model_params = {
        'lr': lr,
        'gamma': gamma,
        'early_stopping_return': None,
        'entropy_reg_factor': entropy_reg_factor,
        'backup_depth': backup_depth,
        'envname': "LunarLander-v2",
        'num_training_steps': num_training_steps,
    }

    finalscores = []
    for _ in range(num_repeats):
        model = modeltype(**model_params) 
        model.train_model()
    
        finalscores.append(model.eval_returns[-1] - model.total_time/model_params['num_training_steps'] * 20)

    return np.mean(finalscores)


def do_study():
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.GPSampler())
    
    study.optimize(lambda trial: objective(trial, modeltype=LunarLanderAC, num_repeats=2, num_training_steps=1e5), n_trials=5)
    
    print("best params:", study.best_params)
    print("best value:", study.best_value)
    with open('trial_results.pk', 'wb+') as f:
        pk.dump(study, f)


def eval_study():
    with open('trial_results.pk', 'rb') as f:
        study = pk.load(f)
    plot_contour(study)
    plt.show()

    
if __name__ == '__main__':
    do_study()
    eval_study()