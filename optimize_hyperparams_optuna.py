import optuna
import numpy as np
from babymode_policybased import PolicyTrainer
import pickle as pk

from optuna.visualization.matplotlib import plot_contour
import matplotlib.pyplot as plt

from model_parameters import ModelParameters


def objective(trial, model_type: str, num_training_steps: int, num_repeats: int) -> float:
    assert model_type in ['actor_critic', 'REINFORCE']

    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", .9, 1)
    entropy_reg_factor = trial.suggest_float("eta", 1e-5, 1e-2, log=True)
    backup_depth = trial.suggest_int("backup_depth", 10, 200, step=5)
    
    model_params = ModelParameters(
        envname = "LunarLander-v2",
        model_type = model_type,
        lr=lr, 
        gamma=gamma, 
        early_stopping_return=None, 
        entropy_reg_factor = entropy_reg_factor,
        backup_depth = backup_depth,
        num_training_steps = num_training_steps,
    )

    finalscores = []
    for _ in range(num_repeats):
        trainer: PolicyTrainer = model_type(model_params) 
        trainer.train_model()
    
        finalscores.append(trainer.final_reward - trainer.total_time/num_training_steps * 20)

    return np.mean(finalscores)


def do_study():
    study = optuna.create_study(direction="maximize", 
                                sampler=optuna.samplers.GPSampler())
    
    study.optimize(lambda trial: objective(trial, model_type='actor_critic', num_repeats=2, num_training_steps=1e5), n_trials=5)
    
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