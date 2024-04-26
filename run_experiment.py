import numpy as np
import matplotlib.pyplot as plt
from babymode_actorcritic import LunarLanderAC

#Parameters
num_repetitions = 3
training_steps = 10_000
model_params = {
            'lr': 0.001,
            'gamma': .99,
            'early_stopping_return': None,
            'entropy_reg_factor': 0.1,
            'backup_depth': 500,
            'envname': "LunarLander-v2",
            'num_training_steps' : training_steps
            }
save_figure = False

for repetition in range(num_repetitions):
    #Initialize reinforcer
    reinforcer = LunarLanderAC(**model_params)

    #Train the reinforcer
    try:
        reinforcer.train_model()
    except KeyboardInterrupt:
        pass
    
    #Initialization of the array of evaluation returns and times
    if repetition == 0:
        all_eval_returns = np.array([reinforcer.eval_returns])
        eval_times = reinforcer.eval_times
    #Appending of the rest of the repetitions to the returns
    else:
        all_eval_returns = np.append(all_eval_returns,[reinforcer.eval_returns],axis=0)
    
    #Calculate mean
    mean_eval_returns = np.mean(all_eval_returns,axis=0)

#Plotting
fig,ax = plt.subplots(figsize=(8,8))

ax.plot(eval_times,mean_eval_returns,label="Mean eval reward over n reps")
param_string = f"Parameters\n$\\alpha$ = {model_params['lr']}\n$\gamma$ = {model_params['gamma']}\nER = {model_params['entropy_reg_factor']}\nBD = {model_params['backup_depth']}"
plt.figtext(0.85,0.85,param_string,bbox=dict(facecolor='black', alpha=0.8, edgecolor='black'),color="white")

ax.grid(alpha=0.5)
ax.set_xlabel("Eval timesteps")
ax.set_ylabel("Mean eval episode reward")
ax.legend()
fig.suptitle(f"{num_repetitions}-repetition averaged evaluation reward")
if save_figure:
    plt.savefig("temp_name.png",dpi=500)
plt.show()
