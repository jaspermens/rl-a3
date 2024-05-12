import numpy as np
import matplotlib.pyplot as plt

from babymode_policybased import PolicyTrainer
from model_parameters import ModelParameters

from argparse import ArgumentParser
import os
    
def train_models(num_repetitions: int, model_params: ModelParameters):
    def pad_early_stopped(arr1: list[int], arr2):
        """pads arr1 to match the length of arr2"""
        npad = len(arr2) - len(arr1)
        padded = np.pad(np.array(arr1), pad_width=(0, npad), mode='constant', constant_values=arr1[-1])
        return padded
    
    eval_times = np.arange(0, model_params.num_training_steps,  model_params.eval_interval)
    eval_returns = np.zeros((num_repetitions, len(eval_times)))
    for i in range(num_repetitions):
        #Initialize reinforcer
        trainer = PolicyTrainer(model_params)

        #Train the reinforcer
        try:
            trainer.train_model()
        except KeyboardInterrupt:
            pass
        
        eval_returns[i] = pad_early_stopped(trainer.eval_returns, eval_times)
        #Calculate mean
        # mean_eval_returns = np.mean(eval_returns,axis=0)
    
    return eval_times, eval_returns


def get_plotdata(num_repetitions: int, model_params: ModelParameters, figname: str = "test"):
    plotdata_fn = f"plotdata_{figname}.npy"
    if os.path.exists(plotdata_fn):
        eval_times, eval_returns = np.load(plotdata_fn, allow_pickle=True)

    else:
        eval_times, eval_returns = train_models(num_repetitions, model_params)
        np.save(plotdata_fn, np.array([eval_times, eval_returns], dtype='object'), allow_pickle=True)        
    
    return eval_times, eval_returns


def plot_training(num_repetitions: int, 
                  model_params: ModelParameters, 
                  figname: str = "test", 
                  save_figure: bool = True):
    eval_times, eval_returns = get_plotdata(num_repetitions=num_repetitions, model_params=model_params, figname=figname)
    
    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot(eval_times,np.mean(eval_returns, axis=0),label="Mean eval reward over n reps")
    ax.fill_between(eval_times, np.min(eval_returns, axis=0), np.max(eval_returns, axis=0), 
                    interpolate=True, alpha=.3, zorder=-1, color='teal',label="Total range")
    
    param_string = f"Parameters\n$\\alpha$ = {model_params.lr}\n$\gamma$ = {model_params.gamma}\nER = {model_params.entropy_reg_factor}\nBD = {model_params.backup_depth}"
    plt.figtext(0.85,0.85,param_string,bbox=dict(facecolor='black', alpha=0.8, edgecolor='black'),color="white")

    ax.grid(alpha=0.5)
    ax.set_xlabel("Eval timesteps")
    ax.set_ylabel("Mean eval episode reward")
    ax.legend()
    fig.suptitle(f"{num_repetitions}-repetition averaged evaluation reward")
    if save_figure:
        plt.savefig("temp_name.png", dpi=500)
    plt.show()


def plot_training_reportstyle(num_repetitions: int, 
                              model_params: ModelParameters, 
                              figname: str = "test",
                              ):
    
    eval_times, eval_returns = get_plotdata(num_repetitions=num_repetitions, model_params=model_params, figname=figname)
    
    fig, ax = plt.subplots(figsize=(8,4.8), 
                          layout="constrained",
                          dpi=150,
                          )

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
    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval return")
    ax.legend()

    plt.savefig(f"figures/{figname}.png")
    plt.savefig(f"figures/{figname}.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    #Parameters
    parser = ArgumentParser(description="""DQN agent training.""")


    parser.add_argument("--filename",
                        dest='filename',
                        type=str,
                        default='test',
                        help="Filename for the learning progression figure."
                        )
    parser.add_argument("--num_training_steps",
                        type=int,
                        default=5e5,
                        help="Max number of training steps."
                        )
    parser.add_argument("--num_repetitions",
                        type=int,
                        default=5,
                        help="Number of experiments to average results over."
                        )
    parser.add_argument("--env",
                        type=str,
                        choices=["CartPole-v1", "LunarLander-v2"],
                        default="LunarLander-v2",
                        help="Name of the Gym environment where the DQN will try to learn."
                        )
    parser.add_argument("--agent_type",
                        type=str,
                        choices=["REINFORCE", "actor_critic"],
                        default="REINFORCE",
                        help="Type of agent to train; either 'REINFORCE' or 'actor_critic'",
                        )    
    parser.add_argument("--do_bootstrap", 
                        dest='do_bootstrap', 
                        action='store_true', 
                        help="""Train with TD bootstrapping (actor-critic only!)."""
                        )
    parser.add_argument("--do_baseline_sub", 
                        dest='do_baseline_sub', 
                        action='store_true', 
                        help="""Train with baseline subtraction (actor-critic only!)."""
                        )
    parser.add_argument("--show_plot",
                        dest='show_plot',
                        action='store_true',
                        help="Run without showing the plot at the end of training. (only saves)",
                        )
    args = parser.parse_args()
    
    model_params = ModelParameters(
                envname = args.env,
                lr = 0.0025,
                gamma = .97,
                entropy_reg_factor = 0.01,
                early_stopping_return = None,
                num_training_steps = args.num_training_steps,
                agent_type = args.agent_type,
                backup_depth = 200,
                do_bootstrap = args.do_bootstrap,
                do_baseline_sub = args.do_baseline_sub,
                )
    
    save_figure = True

    # plot_training(num_repetitions=5, 
    #               model_params=model_params, 
    #               save_figure=save_figure)
    
    plot_training_reportstyle(num_repetitions=args.num_repetitions, 
                  model_params=model_params, 
                  figname=args.filename,
                  )
