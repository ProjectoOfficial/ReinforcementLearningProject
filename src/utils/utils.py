import torch
import numpy as np
import random

def set_seed(env, cfg, verbose):
    if cfg["ppo_random_seed"]:
        if verbose:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", cfg["ppo_random_seed"])
        
        manual_seed = cfg["ppo_random_seed"]
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        env.seed(manual_seed)
        np.random.seed(manual_seed)


def print_hyperparameters(cfg, state_dim, action_dim, run_num, log_f_name, verbose):
    if verbose:
        print("============================================================================================")
        print("training environment name : " + cfg["env_name"])
        
        print("current logging run number for " + cfg["env_name"] + " : ", run_num)
        print("logging at : " + log_f_name)
        
        print("--------------------------------------------------------------------------------------------")
        print("max training timesteps : ", int(cfg["max_training_timesteps"]))
        print("max timesteps per episode : ", int(cfg["max_ep_len"]))
        print("model saving frequency : " + str(int(cfg["save_model_freq"])) + " timesteps")
        print("log frequency : " + str(cfg["max_ep_len"] * cfg["log_freq_multiplier"]) + " timesteps")
        print("printing average reward over episodes in last : " + str(cfg["max_ep_len"] * cfg["print_freq_multiplier"]) + " timesteps")
        print("--------------------------------------------------------------------------------------------")
        print("state space dimension : ", state_dim)
        print("action space dimension : ", action_dim)
        print("--------------------------------------------------------------------------------------------")
        if cfg["has_continuous_action_space"]:
            print("Initializing a continuous action space policy")
            print("--------------------------------------------------------------------------------------------")
            print("starting std of action distribution : ", cfg["action_std"])
            print("decay rate of std of action distribution : ", cfg["action_std_decay_rate"])
            print("minimum std of action distribution : ", cfg["min_action_std"])
            print("decay frequency of std of action distribution : " + str(int(cfg["action_std_decay_freq"])) + " timesteps")
        else:
            print("Initializing a discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("PPO update frequency : " + str(cfg["max_ep_len"] * cfg["ppo_update_timestep_multiplier"]) + " timesteps")
        print("PPO K epochs : ", cfg["ppo_K_epochs"])
        print("PPO epsilon clip : ", cfg["ppo_eps_clip"])
        print("discount factor (gamma) : ", cfg["ppo_gamma"])
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", cfg["ppo_lr_actor"])
        print("optimizer learning rate critic : ", cfg["ppo_lr_critic"])
    