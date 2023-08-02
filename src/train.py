import os
from datetime import datetime
import torch

import cv2
import gym
import json
import argparse

from ppo import PPO
from actors.actor import Actor
from utils.utils import set_seed, print_hyperparameters, build_project, create_videos_from_imgs
from customlogger import CustomLogger

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


def train(cfg: dict, args):
    project_dir = build_project(args.output_path, args.project_name)   
    device = get_device()
    
    env = gym.make(cfg["env_name"], render_mode="rgb_array")
    logger = CustomLogger(os.path.join(project_dir, "logs"))
    set_seed(env, cfg, args.verbose)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if cfg["has_continuous_action_space"] else env.action_space.n
        
    print_hyperparameters(cfg, state_dim, action_dim, os.path.join(project_dir, "logs"), args.verbose)
    ppo_cfg = [cfg["ppo_lr_actor"], cfg["ppo_lr_critic"], cfg["ppo_gamma"], cfg["ppo_K_epochs"], \
            cfg["ppo_eps_clip"], cfg["has_continuous_action_space"], cfg["action_std"]]


    ppo_agent = PPO(Actor, state_dim, action_dim, *ppo_cfg, device=device, process_image=args.process_image)
        
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    running_reward = 0
    running_episodes = 0
    loss, l_clip, l_value_function, l_entropy, state_values, clamped = 0, 0, 0, 0, 0, 0
    for ep in range(int(cfg["episodes"])):
        state = {
            "env_state": env.reset()[0],
            "env_image": env.render() if args.process_image else None
        }
        current_ep_reward = 0

        #Training Loop for a single episode (or until we reach the max number ot iterations for a single episode)
        for t in range(cfg["episode_length"]):

            # select action with policy and perform the selected action on the environment
            action = ppo_agent.select_action(state)
            new_state, reward, done, _, _ = env.step(action)
            
            state["env_state"] = new_state
            state["env_image"] = env.render() if args.process_image else None

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step = (ep *  int(cfg["episode_length"])) + t
            current_ep_reward += reward

            # update PPO agent
            # since we use Monte Carlo for the updates, we wait until we have sampled N rewards and then we train
            # actor's and critic's neural networks
            if time_step % (cfg["episode_length"] * cfg["ppo_update_timestep_multiplier"]) == 0 and time_step != 0:
                loss, l_clip, l_value_function, l_entropy, state_values, clamped = ppo_agent.update()
                
            # if continuous action space; then decay action std of ouput action distribution
            if cfg["has_continuous_action_space"] and time_step % int(cfg["action_std_decay_freq"]) == 0 and time_step != 0:
                ppo_agent.decay_action_std(cfg["action_std_decay_rate"], cfg["min_action_std"])

            # log in logging file
            if time_step % (cfg["episode_length"] * cfg["log_freq_multiplier"]) == 0 and time_step != 0:
                avg_reward = running_reward / running_episodes
                avg_reward = round(avg_reward, 4)

                logger.log(time_step, avg_reward, ep, loss, l_clip, l_value_function, l_entropy, state_values, clamped)
                
                running_reward = 0
                running_episodes = 0

            # save model weights
            if time_step % int(cfg["save_model_freq"]) == 0 and time_step != 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = os.path.join(project_dir, "ckpts", f"{time_step}.pth")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if args.render or ep == 0 or (ep % cfg["save_img_freq"] == 0):
                frame = env.render()
                
                if ep == 0 or ep % cfg["save_img_freq"] == 0:
                    im_path = os.path.join(project_dir, "imgs", f"{ep}")
                    if not os.path.exists(im_path):
                        os.makedirs(im_path)
                    
                    cv2.imwrite(os.path.join(im_path, f"{t}.jpg" ), frame)
                
                if args.render:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1000//30) # 30fps

            # break; if the episode is over
            if done:
                break

        running_reward += current_ep_reward
        running_episodes += 1

    logger.close()
    env.close()
    
    create_videos_from_imgs(os.path.join(project_dir, "imgs"))

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
    parser.add_argument("--output-path", type=str, help="output directory")
    parser.add_argument("--project-name", type=str, help="project name")    
    parser.add_argument("--pretrained-run", type=int, default=0, help="set this to load a particular checkpoint num")
    args = parser.parse_args()
    
    assert args.cfg_path is not None and os.path.isfile(args.cfg_path), "cfg_path is not specified or is not a json file absolute path"
    
    cfg = read_cfg(args.cfg_path)
    train(cfg, args)
