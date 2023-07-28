import os
from datetime import datetime
import torch

import cv2
import gym
import json
import argparse

from ppo import PPO
from actors.actor import Actor
from utils.utils import set_seed, print_hyperparameters


def read_cfg(cfg_path):
    cfg = None
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    return cfg


def get_device():
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    return device


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train(cfg: dict, args):
    device = get_device()
    
    env = gym.make(cfg["env_name"], render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if cfg["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # log files for multiple runs are NOT overwritten
    create_path(args.log_path)
    create_path(os.path.join(args.log_path, cfg["env_name"]))

    # get number of log files in log directory
    current_num_files = next(os.walk(args.log_path))[2]
    run_num = len(current_num_files)
    log_f_name = os.path.join(args.log_path, 'PPO_' + cfg["env_name"] + "_log_" + str(run_num) + ".csv")

    create_path(args.ckpt_path)
    create_path(os.path.join(args.ckpt_path, cfg["env_name"]))
    
    checkpoint_path = os.path.join(args.ckpt_path, cfg["env_name"], "PPO_{}_{}_{}.pth".format(cfg["env_name"], cfg["ppo_random_seed"], args.pretrained_run))
    print("save checkpoint path : " + checkpoint_path)
    
    set_seed(env, cfg, args.verbose)
    print_hyperparameters(cfg, state_dim, action_dim, run_num, log_f_name, args.verbose)
    ppo_cfg = [cfg["ppo_lr_actor"], cfg["ppo_lr_critic"], cfg["ppo_gamma"], cfg["ppo_K_epochs"], \
            cfg["ppo_eps_clip"], cfg["has_continuous_action_space"], cfg["action_std"]]

    # initialize a PPO agent
    ppo_agent = PPO(Actor, state_dim, action_dim, *ppo_cfg, device=device, process_image=args.process_image)
        
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
    running_loss = 0

    # training loop
    while time_step <= int(cfg["max_training_timesteps"]):
        state = {
            "env_state": env.reset()[0],
            "env_image": env.render() if args.process_image else None
        }
        current_ep_reward = 0

        #Training Loop for a single episode (or until we reach the max number ot iterations for a single episode)
        for t in range(1, cfg["max_ep_len"] + 1):

            # select action with policy and perform the selected action on the environment
            action = ppo_agent.select_action(state)
            new_state, reward, done, _, _ = env.step(action)
            
            state["env_state"] = new_state
            state["env_image"] = env.render() if args.process_image else None

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            # since we use Monte Carlo for the updates, we wait until we have sampled N rewards and then we train
            # actor's and critic's neural networks
            if time_step % (cfg["max_ep_len"] * cfg["ppo_update_timestep_multiplier"]) == 0:
                running_loss = ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if cfg["has_continuous_action_space"] and time_step % int(cfg["action_std_decay_freq"]) == 0:
                ppo_agent.decay_action_std(cfg["action_std_decay_rate"], cfg["min_action_std"])

            # log in logging file
            if time_step % (cfg["max_ep_len"] * cfg["log_freq_multiplier"]) == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % (cfg["max_ep_len"] * cfg["print_freq_multiplier"]) == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Loss : {:.5f}".format(i_episode, time_step, print_avg_reward, running_loss))

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

            if args.render:
                frame = env.render()
                cv2.imshow('frame', frame)
                cv2.waitKey(1000//30) # 30fps

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
    parser.add_argument("--process-image", action="store_true", help="use also game image as state observation")
    parser.add_argument("--verbose", action="store_true", help="verbosity")
    parser.add_argument("--render", action="store_true", help="show environment progress")
    parser.add_argument("--log-path", type=str, help="absolute path where to store logs")
    parser.add_argument("--ckpt-path", type=str, help="absolute path where to store checkpoints")    
    parser.add_argument("--pretrained-run", type=int, default=0, help="set this to load a particular checkpoint num")
    args = parser.parse_args()
    
    assert args.cfg_path is not None and os.path.isfile(args.cfg_path), "cfg_path is not specified or is not a json file absolute path"
    
    cfg = read_cfg(args.cfg_path)
    train(cfg, args)
