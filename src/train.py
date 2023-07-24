import os
from datetime import datetime
import torch
import numpy as np

import gym
import json
import argparse

from ppo import PPO


def read_cfg(cfg_path):
    cfg = None
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    return cfg


################################### Training ###################################
def train(cfg: dict):
    print("============================================================================================")
    print("training environment name : " + cfg["env_name"])

    env = gym.make(cfg["env_name"])

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if cfg["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = os.path.join(log_dir, cfg["env_name"])
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = os.path.join(log_dir, 'PPO_' + cfg["env_name"] + "_log_" + str(run_num) + ".csv")

    print("current logging run number for " + cfg["env_name"] + " : ", run_num)
    print("logging at : " + log_f_name)

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = os.path.join(directory, cfg["env_name"])
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(cfg["env_name"], cfg["ppo_random_seed"], run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    ############# print all hyperparameters #############
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
    if cfg["ppo_random_seed"]:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", cfg["ppo_random_seed"])
        torch.manual_seed(cfg["ppo_random_seed"])
        env.seed(cfg["ppo_random_seed"])
        np.random.seed(cfg["ppo_random_seed"])
    print("============================================================================================")

    ################# training procedure ################
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, cfg["ppo_lr_actor"], cfg["ppo_lr_critic"], cfg["ppo_gamma"], cfg["ppo_K_epochs"], \
        cfg["ppo_eps_clip"], cfg["has_continuous_action_space"], cfg["action_std"])

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= int(cfg["max_training_timesteps"]):

        state = env.reset()[0]
        current_ep_reward = 0

        #Training Loop for a single episode (or until we reach the max number ot iterations for a single episode)
        for t in range(1, cfg["max_ep_len"] + 1):

            # select action with policy and perform the selected action on the environment
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            # since we use Monte Carlo for the updates, we wait until we have sampled N rewards and then we train
            # actor's and critic's neural networks
            if time_step % (cfg["max_ep_len"] * cfg["ppo_update_timestep_multiplier"]) == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if cfg["has_continuous_action_space"] and time_step % int(cfg["action_std_decay_freq"]) == 0:
                ppo_agent.decay_action_std(cfg["action_std_decay_rate"], cfg["min_action_std"])

            # log in logging file
            if time_step % (cfg["max_ep_len"] * cfg["log_freq_multiplier"]) == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % (cfg["max_ep_len"] * cfg["print_freq_multiplier"]) == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % int(cfg["save_model_freq"]) == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="json configuration file path containing environment and ppo config")
    args = parser.parse_args()
    
    assert args.cfg_path is not None and os.path.isfile(args.cfg_path), "cfg_path is not specified or is not a json file absolute path"
    
    cfg = read_cfg(args.cfg_path)
    train(cfg)
