import torch
import numpy as np
import random
import os
import subprocess
import json

from gym.wrappers.time_limit import TimeLimit

def read_cfg(cfg_path: str):
    cfg = None
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    cfg_sanity_check(cfg)
        
    return cfg


def cfg_sanity_check(cfg: dict):
    
    general_sanity_check(cfg, "env_name", str, "environment name")
    general_sanity_check(cfg, "has_continuous_action_space", bool, "has_continuous_action_space")
    
    general_sanity_check(cfg, "episodes", int, "the number of iterations within an episodes")
    
    general_sanity_check(cfg, "episodes", int, "the number of episodes")
    
    general_sanity_check(cfg, "save_img_freq", int, "save image")
    general_sanity_check(cfg, "save_model_freq", int, "save model")
    general_sanity_check(cfg, "log_freq", int, "logging")
    
    general_sanity_check(cfg, "action_std", float, "the action standard deviation", False)
    general_sanity_check(cfg, "action_std_decay_rate", float, "the action standard deviation decay", False)
    general_sanity_check(cfg, "min_action_std", float, "the minimum action standard deviation", False)
    general_sanity_check(cfg, "action_std_decay_freq", int, "the action standard deviation decay frequency", False)
    
    general_sanity_check(cfg, "ppo_update", int, "PPO update")
    general_sanity_check(cfg, "ppo_K_epochs", int, "PPO K epochs")
    general_sanity_check(cfg, "ppo_eps_clip", float, "PPO eps clip")
    general_sanity_check(cfg, "ppo_gamma", float, "PPO gamma")
    general_sanity_check(cfg, "ppo_lr_actor", float, "PPO actor learning rate")
    general_sanity_check(cfg, "ppo_lr_critic", float, "PPO critic learning rate")
    general_sanity_check(cfg, "ppo_random_seed", int, "PPO random seed")
    general_sanity_check(cfg, "ppo_c1", float, "PPO C1 parameter")
    general_sanity_check(cfg, "ppo_c2", float, "PPO C2 parameter")
    general_sanity_check(cfg, "ppo_lamb", float, "PPO lambda parameter")

    general_sanity_check(cfg, "network_activations", str, "network hidden activation functions")
    general_sanity_check(cfg, "network_last_activation", str, "network last activation function")
    general_sanity_check(cfg, "network_hiddel_layers", list, "network hidden layers list")
    
    
def general_sanity_check(cfg: dict, key: str, class_type, what: str, never_none: bool=True):
    assert key in cfg.keys(), f"{what} must be specified"
    
    if never_none or cfg[key] is not None:
        assert isinstance(cfg[key], class_type), f"{what} must be an instance of {class_type}"
        
        if class_type == int or class_type == float:
            assert cfg[key] > 0, f"{what} must be strictly positive"
            
        if class_type == str:
            assert cfg[key] != "", f"{what} must not be empty"
            
        if class_type == list:
            assert cfg[key] != [], f"{what} must not be empty"
            

def get_device():
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    return device


def set_seed(env: TimeLimit, cfg: dict, verbose: dict):
    if cfg["ppo_random_seed"]:
        if verbose:
            print("--------------------------------------------------------------------------------------------")
            print("setting random seed to ", cfg["ppo_random_seed"])
        
        manual_seed = cfg["ppo_random_seed"]
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(manual_seed)
        if getattr(env, "seed", None):
            env.seed(manual_seed)
        np.random.seed(manual_seed)


def create_path(path: str):
    assert path is not None and path != "", "Ensure that the path you want to use is specified as parameter"
    if not os.path.exists(path):
        os.makedirs(path)


def build_project(output_dir: str, project_name: str):
    create_path(output_dir)
    create_path(os.path.join(output_dir, project_name))
    create_path(os.path.join(output_dir, project_name, "logs"))
    create_path(os.path.join(output_dir, project_name, "ckpts"))
    create_path(os.path.join(output_dir, project_name, "imgs"))
    return os.path.join(output_dir, project_name)


def create_videos_from_imgs(root_path: str, fps: int=30):
    for subfolder in os.listdir(root_path):
        path = os.path.join(root_path, subfolder)
        for dirpath, _, filenames in os.walk(path):
            video_name = os.path.basename(dirpath) + '.mp4'
            video_path = os.path.join(path, video_name)

            img_files = [os.path.join(dirpath, f) for f in filenames if f.lower().endswith('.jpg')]

            img_files.sort()

            if img_files:
                ffmpeg_cmd = f"ffmpeg -framerate {fps} -i %d.jpg -c:v libx264 -r {fps} {video_path}"
                subprocess.run(ffmpeg_cmd, shell=True, cwd=dirpath)


def print_hyperparameters(cfg: dict, state_dim: int, action_dim: int, log_f_name: str, verbose: bool):
    if verbose:
        print("============================================================================================")
        print("training environment name : " + cfg["env_name"])
        print("logging at : " + log_f_name)
        
        print("--------------------------------------------------------------------------------------------")
        print("episodes : ", int(cfg["episodes"]))
        print("max timesteps per episode : ", int(cfg["episode_length"]))
        print("model saving frequency : " + str(int(cfg["save_model_freq"])))
        print("log frequency : " + str(cfg["log_freq"]))
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
        print("PPO update frequency : " + str(cfg["episode_length"] * cfg["ppo_update"]))
        print("PPO K epochs : ", cfg["ppo_K_epochs"])
        print("PPO epsilon clip : ", cfg["ppo_eps_clip"])
        print("discount factor (gamma) : ", cfg["ppo_gamma"])
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", cfg["ppo_lr_actor"])
        print("optimizer learning rate critic : ", cfg["ppo_lr_critic"])
    