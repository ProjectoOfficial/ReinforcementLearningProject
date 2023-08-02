import os
import json
import argparse
import gym
import cv2

from ppo import PPO
from actors.actor import Actor


def read_cfg(cfg_path):
    cfg = None
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    return cfg


def test(cfg, args):
    env = gym.make(cfg["env_name"], render_mode="rgb_array")

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if cfg["has_continuous_action_space"]:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(Actor, state_dim, action_dim, cfg["ppo_lr_actor"], cfg["ppo_lr_critic"] , cfg["ppo_gamma"], \
        cfg["ppo_K_epochs"], cfg["ppo_eps_clip"], cfg["has_continuous_action_space"], cfg["action_std"])

    ckpt_path = os.path.join(args.output_path, args.project_name, "ckpts", str(args.ckpt_number) + ".pth")
    assert os.path.isfile(ckpt_path), f"this file does not exists: {ckpt_path}"
    print("loading network from : " + ckpt_path)

    ppo_agent.load(ckpt_path)
    test_running_reward = 0
    
    for ep in range(args.test_episodes):
        ep_reward = 0
        state = {
            "env_state": env.reset()[0],
            "env_image": env.render() if args.process_image else None
        }

        for t in range(cfg["episode_length"]):
            action = ppo_agent.select_action(state)
            new_state, reward, done, _ = env.step(action)[0:-1]
            ep_reward += reward
            
            state["env_state"] = new_state
            state["env_image"] = env.render() if args.process_image else None

            if args.render:
                frame = env.render()
                cv2.imshow('frame', frame)
                cv2.waitKey(1000//30) # 30fps

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0
        
    cv2.destroyAllWindows()
    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / args.test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", type=str, help="json configuration file path containing environment and ppo config")
    parser.add_argument("--render", action="store_true", help="render the videogame or just run silently (default: no render)")
    parser.add_argument("--output-path", type=str, help="output directory")
    parser.add_argument("--project-name", type=str, help="project name")    
    parser.add_argument("--ckpt-number", type=int, help="checkpoint number")    
    parser.add_argument("--test-episodes", type=int, default=10, help="total num of testing episodes")
    parser.add_argument("--process-image", action="store_true", help="use also game image as state observation")
    args = parser.parse_args()
    
    assert args.cfg_path is not None and os.path.isfile(args.cfg_path), "cfg_path is not specified or is not a json file absolute path"
    assert args.test_episodes > 0, "Cannot set less than 1 episodes"
    
    cfg = read_cfg(args.cfg_path)
    
    test(cfg, args)
