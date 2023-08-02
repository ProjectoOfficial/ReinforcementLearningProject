import torch
import numpy as np
import random
import os
import subprocess

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


def create_path(path):
    assert path is not None and path != "", "Ensure that the path you want to use is specified as parameter"
    if not os.path.exists(path):
        os.makedirs(path)


def build_project(output_dir, project_name):
    create_path(output_dir)
    create_path(os.path.join(output_dir, project_name))
    create_path(os.path.join(output_dir, project_name, "logs"))
    create_path(os.path.join(output_dir, project_name, "ckpts"))
    create_path(os.path.join(output_dir, project_name, "imgs"))
    return os.path.join(output_dir, project_name)


def create_videos_from_imgs(root_path, fps=30):
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


def print_hyperparameters(cfg, state_dim, action_dim, log_f_name, verbose):
    if verbose:
        print("============================================================================================")
        print("training environment name : " + cfg["env_name"])
        print("logging at : " + log_f_name)
        
        print("--------------------------------------------------------------------------------------------")
        print("episodes : ", int(cfg["episodes"]))
        print("max timesteps per episode : ", int(cfg["episode_length"]))
        print("model saving frequency : " + str(int(cfg["save_model_freq"])) + " timesteps")
        print("log frequency : " + str(cfg["episode_length"] * cfg["log_freq_multiplier"]) + " timesteps")
        print("printing average reward over episodes in last : " + str(cfg["episode_length"] * cfg["print_freq_multiplier"]) + " timesteps")
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
        print("PPO update frequency : " + str(cfg["episode_length"] * cfg["ppo_update_timestep_multiplier"]) + " timesteps")
        print("PPO K epochs : ", cfg["ppo_K_epochs"])
        print("PPO epsilon clip : ", cfg["ppo_eps_clip"])
        print("discount factor (gamma) : ", cfg["ppo_gamma"])
        print("--------------------------------------------------------------------------------------------")
        print("optimizer learning rate actor : ", cfg["ppo_lr_actor"])
        print("optimizer learning rate critic : ", cfg["ppo_lr_critic"])
    