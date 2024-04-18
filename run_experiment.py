from policies import Policy
from argparse import ArgumentParser 



if __name__ == "__main__":
    parser = ArgumentParser(description="""DQN agent training.""")

    parser.add_argument("--no_target_network", 
                        dest='target_network', 
                        action='store_false', 
                        help="""Train without a target network.""")
    parser.add_argument("--no_experience_replay", 
                        dest='experience_replay', 
                        action='store_false', 
                        help="""Train without experience replay.""")
    parser.add_argument("--filename",
                        dest='filename',
                        type=str,
                        default='test_results',
                        help="Filename for the learning progression figure.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        help="Max number of training episodes.")
    parser.add_argument("--num_repetitions",
                        type=int,
                        default=20,
                        help="Number of experiments to average results over.")
    parser.add_argument("--env",
                        type=str,
                        choices=["CartPole-v1", "CartPole-v0", "LunarLander-v2"],
                        default="CartPole-v1",
                        help="Name of the Gym environment where the DQN will try to learn."
                        )
    parser.add_argument("--policy",
                        type=str,
                        choices=["egreedy", "softmax"],
                        default="softmax",
                        help="Exploration strategy. Either epsilon-greedy or softmax/boltzmann",
                        )    
    parser.add_argument("--show_plot",
                        dest='show_plot',
                        action='store_true',
                        help="Run without showing the plot at the end of training. (only saves)",
                        )
    cmdargs = parser.parse_args()
    
    policy_param_annealtime = {
        "egreedy": (Policy.EGREEDY, 0.1, 100),
        "softmax": (Policy.SOFTMAX, 0.5, 50),
    }

    policy, exp_param, anneal_timescale = policy_param_annealtime[cmdargs.policy] 

    model_params = {
            'lr': 0.01,  
            'exp_param': exp_param,
            'policy': policy, 
            'batch_size': 256,
            'gamma': .995,
            'target_network_update_time': 50,
            'do_target_network': cmdargs.target_network,
            'do_experience_replay': cmdargs.experience_replay,
            'buffer_capacity': 10000, 
            'eval_interval': 1,
            'n_eval_episodes': 20,
            'anneal_exp_param': False,
            'anneal_timescale': anneal_timescale,
            'early_stopping_reward': 500 if cmdargs.env=="CartPole-v1" else None,
    }
    make_learning_plots(
        num_epochs = cmdargs.num_epochs, 
        num_repetitions = cmdargs.num_repetitions, 
        model_params = model_params,
        filename = cmdargs.filename,
        environment_name = cmdargs.env,
        show_plot=cmdargs.show_plot,
        )