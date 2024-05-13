from run_experiment import get_plotdata
from model_parameters import ModelParameters
import matplotlib.pyplot as plt
import numpy as np

def make_comparison_plot():
    experiment_titles = ['REINFORCE', 'AC with bootstrapping', 'AC with baseline subtraction', 'AC with both']
    filenames = ['reinforce', 'ac_bootstrap', 'ac_baselinesub', 'ac_both']
    
    base_params = {'envname': "LunarLander-v2",
                    'lr': 2.5e-3,
                    'gamma': .97,
                    'entropy_reg_factor': 0.01,
                    'early_stopping_return': None,
                    'backup_depth': 200,
                    'eval_interval': 2000,
                    'n_eval_episodes': 20,
                    'num_training_steps': 5e5,
                    }
    
    experiment_params = [
        ['REINFORCE', False, False],
        ['actor_critic', True, False],
        ['actor_critic', False, True],
        ['actor_critic', True, True],
    ]

    hparam_sets = [
        ModelParameters(agent_type=agent_type, do_bootstrap=do_bootstrap, do_baseline_sub=do_baseline_sub, **base_params)
        for agent_type, do_bootstrap, do_baseline_sub in experiment_params
    ]

    fig, axes = plt.subplots(2,2, figsize=(12,6), dpi=200, sharex=True, sharey=True, layout='constrained')

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
        ax.set_title(experiment_title)

    axes[0,0].legend()
    axes[1,0].set_xlabel("Training Step")
    axes[1,1].set_xlabel("Training Step")
    axes[0,0].set_ylabel("Eval Return")
    axes[1,0].set_ylabel("Eval Return")
    
        
    plt.savefig(f"figures/quadplot.png")
    plt.savefig(f"figures/quadplot.pdf")
    plt.close()


if __name__ == "__main__":
    make_comparison_plot()