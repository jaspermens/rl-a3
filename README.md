<h1 align="center">RL A3 - Group 46</h1>

### Assignment 3 submission for the course Reinforcement Learning 2023/2024 at Leiden University

Contains all the code necessary to compare REINFORCE to various actor-critic implementations in the `LunarLander-v2` gym environment

### Code dependencies:

`torch` `tqdm` `gymnasium`

#### Usage:

After cloning the git repo, each experiment can be run with a single command as follows:

##### Note: make sure both a **figures** and a **results** directory exists!

To train the REINFORCE model:

### `python3 run_experiment.py --filename='reinforce' --agent_type 'REINFORCE'`

To train AC with bootstrapping:

### `python3 run_experiment.py --filename='ac_bootstrap' --do_bootstrap --agent_type 'actor_critic'`

To train AC with baseline subtraction:

### `python3 run_experiment.py --filename='ac_baselinesub' --do_baseline_sub --agent_type 'actor_critic'`

To train AC with bootstrapping and baseline subtraction:

### `python3 run_experiment.py --filename='ac_both' --do_bootstrap --do_baseline_sub --agent_type 'actor_critic'`



Each of the above commands will produce a plot of the learning progression. To facilitate comparison, a 2x2 learning plot can be generated  from the stored data by running:

### `python3 make_quadplot.py`

If no stored data is available, new models will be trained. 





For a more complete run-down of the possible command-line parameters, run

### `python3 run_experiment.py --help`

###### Note: previous results will be used by default if they are available. To fully re-run the experiments, you may want to delete the contents of the results directory first.
