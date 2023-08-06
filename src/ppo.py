import torch
import torch.nn as nn
from actors.actor import Actor

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

# from torchrl.objectives import ClipPPOLoss
class PPO:
    def __init__(self, 
                 actor_net: Actor, 
                 state_dim: int, 
                 action_dim: int, 
                 cfg: dict, 
                 device: torch.device="cpu", 
                 process_image: bool=False) -> None:
        self.gamma = cfg["ppo_gamma"]
        self.eps_clip = cfg["ppo_eps_clip"]
        self.K_epochs = cfg["ppo_K_epochs"]
        self.device = device
        self.action_std = cfg["action_std"]
        self.has_continuous_action_space = cfg["has_continuous_action_space"]

        self.buffer = RolloutBuffer()

        self.policy = actor_net(cfg["network_hiddel_layers"], cfg["network_activations"], cfg["network_last_activation"], \
            state_dim, action_dim, cfg["has_continuous_action_space"], cfg["action_std"], device, process_image).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': cfg["ppo_lr_actor"]},
                        {'params': self.policy.critic.parameters(), 'lr': cfg["ppo_lr_critic"]}
                    ])

        self.policy_old = actor_net(cfg["network_hiddel_layers"], cfg["network_activations"], cfg["network_last_activation"], \
            state_dim, action_dim, cfg["has_continuous_action_space"], cfg["action_std"], device, process_image).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.criterion = nn.SmoothL1Loss(reduction="none")
        self.c1 = cfg["ppo_c1"]
        self.c2 = cfg["ppo_c2"]
        self.lamb = cfg["ppo_lamb"]


    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            raise ValueError("WARNING : Calling PPO::set_action_std() on discrete action space policy")


    def decay_action_std(self, action_std_decay_rate: int, min_action_std: float):
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            raise ValueError("WARNING : Calling PPO::decay_action_std() on discrete action space policy")


    def select_action(self, state: dict):                
        if self.has_continuous_action_space:
            with torch.no_grad():
                env_state, action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(env_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                env_state, action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(env_state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    #compute returns estimates, either with MC or with GAE (which uses TD)
    #def compute_estimates(self, mode:str):
    #    if str.lower(mode) == "mc":
    #        # Monte Carlo estimate of returns
    #        rewards = []
    #        discounted_reward = 0
    #        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
    #            if is_terminal:
    #                discounted_reward = 0
    #            discounted_reward = reward + (self.gamma * discounted_reward)
    #            rewards.insert(0, discounted_reward)
    #        # now we have in position 0 the Return G_t
    #    elif str.lower(mode) == "gae":
    #        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
    #            if is_terminal:
    #                discounted_reward = 0
    #            discounted_reward = reward + (self.gamma * discounted_reward)
    #            rewards.insert(0, discounted_reward)
    #            delta = reward + self.gamma * discounted_reward #missing - values, it's performed afterwards
    #            adv = delta
    
    def update(self):
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        # calculate advantages
        advantages = rewards - old_state_values

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO, composed by: 
            # 1) L^CLIP loss (surrogate loss), used to limit the amount of change to the current policy
            # 2) l^VF loss, loss over our value function and target value functions (sampled rewards)
            # 3) S[pi_theta], entropy term introduced to ensure exploration
            # all signs are reversed because we use the optimizer to perform gradient descent
            # instead of using directly gradient ascent
            l_clip = -torch.min(surr1, surr2)
            l_value_function = self.c1 * self.criterion(state_values, rewards)
            l_entropy = - self.c2 * dist_entropy
            loss =  (l_clip + l_value_function + l_entropy).mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        clamped = ((ratios * advantages) == surr2).all().to(torch.int8)
        return loss.mean().detach(), l_clip.mean().detach(), l_value_function.mean().detach(), l_entropy.mean().detach(), state_values.mean().detach(), clamped.detach()
    
    
    def save(self, checkpoint_path: str):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
   
    def load(self, checkpoint_path: str):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


