import os
from datetime import datetime
import torch

import cv2
import gym
import argparse

from ppo import PPO
from actors.actor import Actor
from utils.utils import get_device, read_cfg, set_seed, print_hyperparameters, build_project, create_videos_from_imgs
from customlogger import CustomLogger


def train(cfg: dict, args: argparse.Namespace):
    project_dir = build_project(args.output_path, args.project_name)   
    device = get_device()
    
    env = gym.make(cfg["env_name"], render_mode="rgb_array")
    logger = CustomLogger(os.path.join(project_dir, "logs"))
    set_seed(env, cfg, args.verbose)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if cfg["has_continuous_action_space"] else env.action_space.n
        
    print_hyperparameters(cfg, state_dim, action_dim, os.path.join(project_dir, "logs"), args.verbose)
    ppo_agent = PPO(Actor, state_dim, action_dim, cfg, device=device, process_image=args.process_image)
        
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    running_reward = 0
    running_episodes = 0
    loss, l_clip, l_value_function, l_entropy, state_values, clamped = 0, 0, 0, 0, 0, 0
    for ep in range(int(cfg["episodes"])):
        state = {
            "env_state": torch.FloatTensor(env.reset()[0]).to(device),
            "env_image": torch.FloatTensor(env.render().copy()).to(device)  if args.process_image else None
        }
        current_ep_reward = 0

        for t in range(cfg["episode_length"]):

            # select action with policy and perform the selected action on the environment
            action = ppo_agent.select_action(state)
            new_state, reward, done, _, _ = env.step(action)
            
            state["env_state"] = torch.FloatTensor(new_state).to(device)
            state["env_image"] = torch.FloatTensor(env.render().copy()).to(device)  if args.process_image else None

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step = (ep *  int(cfg["episode_length"])) + t
            current_ep_reward += reward

            if args.render or ((ep + 1) % cfg["save_img_freq"] == 0):
                frame = env.render()
                
                if (ep + 1) % cfg["save_img_freq"] == 0:
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
        
        # update PPO agent
        # since we use Monte Carlo for the updates, we wait until we have sampled N rewards and then we train
        # actor's and critic's neural networks
        if ((ep + 1) % cfg["ppo_update"] == 0):
            loss, l_clip, l_value_function, l_entropy, state_values, clamped = ppo_agent.update()
        
        
        # if continuous action space; then decay action std of ouput action distribution
        if cfg["has_continuous_action_space"] and ((ep + 1) % cfg["action_std_decay_freq"] == 0):
            ppo_agent.decay_action_std(cfg["action_std_decay_rate"], cfg["min_action_std"])
                
        
        if ep > 0 and ((ep + 1) % cfg["log_freq"] == 0):
                avg_reward = running_reward / running_episodes if running_episodes else 0
                avg_reward = round(avg_reward, 4)

                logger.log(time_step, avg_reward, ep, loss, l_clip, l_value_function, l_entropy, state_values, clamped)
                
                running_reward = 0
                running_episodes = 0

        if ep > 0 and ((ep + 1) % cfg["save_model_freq"] == 0):
            print("--------------------------------------------------------------------------------------------")
            checkpoint_path = os.path.join(project_dir, "ckpts", f"{ep}.pth")
            ppo_agent.save(checkpoint_path)
            print(f"Saving: {checkpoint_path} Elapsed Time  : {datetime.now().replace(microsecond=0) - start_time}")
            print("--------------------------------------------------------------------------------------------")


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
    parser.add_argument("output_path", type=str, help="output directory")
    parser.add_argument("project_name", type=str, help="project name")    
    parser.add_argument("--process-image", action="store_true", help="use also game image as state observation")
    parser.add_argument("--verbose", action="store_true", help="verbosity")
    parser.add_argument("--render", action="store_true", help="show environment progress")
    parser.add_argument("--pretrained-run", type=int, default=0, help="set this to load a particular checkpoint num")
    args = parser.parse_args()
    
    assert args.cfg_path is not None and os.path.isfile(args.cfg_path), "cfg_path is not specified or is not a json file absolute path"
    assert args.output_path is not None and os.path.isabs(args.output_path), "the output path must be absolute and must be specified"
    
    cfg = read_cfg(args.cfg_path)
    train(cfg, args)
