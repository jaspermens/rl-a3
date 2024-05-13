 
from run_experiment import get_plotdata
from model_parameters import ModelParameters
import matplotlib.pyplot as plt
import numpy as np

def make_eta_plot():
    experiment_titles = ['High $\eta$', 'Optimal $\eta$', 'Low $\eta$']
    filenames = ['high_eta', 'ac_both', 'low_eta']

    base_params = {'envname': "LunarLander-v2",
                    'lr': 2.5e-3,
                    'gamma': .97,
                    'agent_type': 'actor_critic',
                    'do_bootstrap': True,
                    'do_baseline_sub': True,
                    'early_stopping_return': None,
                    'backup_depth': 200,
                    'eval_interval': 2000,
                    'n_eval_episodes': 20,
                    'num_training_steps': 5e5,
                    }
    
    experiment_params = [0.001, 0.01, 0.1]

    hparam_sets = [
        ModelParameters(entropy_reg_factor=eta, **base_params)
        for eta in experiment_params
    ]

    fig, axes = plt.subplots(3,1, figsize=(6,8), dpi=200, sharex=True, sharey=True, layout='constrained')

    for ax, filename, experiment_title, model_params in zip(axes.flatten(), filenames, experiment_titles, hparam_sets):
        eval_times, eval_returns = get_plotdata(num_repetitions=5, model_params=model_params, filename=filename)

        ax.xaxis.set_tick_params(direction='in')
        ax.yaxis.set_tick_params(direction='in')

        ax.plot(eval_times, np.mean(eval_returns, axis=0), color='teal', label='mean')
        
        ax.fill_between(eval_times, 
                        np.min(eval_returns, axis=0), 
                        np.max(eval_returns, axis=0), 
                        interpolate=True, 
                        alpha=.3, 
                        zorder=-1, 
                        color='teal',
                        label="Total range",
                        )
        
        ax.grid(alpha=0.5)
        ax.set_ylabel("Eval Return")
        ax.set_title(experiment_title)

    ax.set_ylim(bottom=-300)
    axes[0].legend()
    axes[-1].set_xlabel("Training Step")
    # axes[1].set_xlabel("Training Step")
    # axes[1].set_ylabel("Eval Return")
    
        
    plt.savefig(f"figures/etaplot.png")
    plt.savefig(f"figures/etaplot.pdf")
    plt.close()


if __name__ == "__main__":
    make_eta_plot()