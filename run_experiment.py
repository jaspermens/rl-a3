import numpy as np
import matplotlib.pyplot as plt

from babymode_policybased import PolicyTrainer
from model_parameters import ModelParameters

    
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


def plot_training(num_repetitions: int, model_params: ModelParameters, save_figure: bool = True):
    eval_times, eval_returns = train_models(num_repetitions, model_params)
    fig,ax = plt.subplots(figsize=(8,8))

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
        plt.savefig("temp_name.png",dpi=500)
    plt.show()


if __name__ == "__main__":
    #Parameters
    training_steps = 2e5
    model_params = ModelParameters(**{
                'envname': "LunarLander-v2",
                'lr': 0.002,
                'gamma': .97,
                'entropy_reg_factor': 0.001226,
                'early_stopping_return': None,
                'num_training_steps' : training_steps,
                'agent_type': 'actor_critic',
                'backup_depth': 50,
                })
    
    save_figure = True

    plot_training(num_repetitions=5, 
                  model_params=model_params, 
                  save_figure=save_figure)
