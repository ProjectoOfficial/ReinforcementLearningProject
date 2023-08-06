import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from actors.actor_utils import build_sequential_network, get_act_from_string
from torchvision import models


class Actor(nn.Module):
    def __init__(self, 
                 net_hiddens: list, 
                 act_funs: str, 
                 last_act_fun: str, 
                 state_dim: int, 
                 action_dim: int, 
                 has_continuous_action_space: bool, 
                 action_std_init: float, 
                 device: torch.device, 
                 process_image: bool) -> None:
        super(Actor, self).__init__()

        self.process_image = process_image
        self.device = device
        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
            
        if process_image:
            self.backbone = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # keep until last conv layer
            
            dummy_input = torch.randn(1, 3, 224, 224)
            state_dim += torch.flatten(self.backbone(dummy_input)).shape[0]
        
        acts = get_act_from_string(act_funs)
        last_act = get_act_from_string(last_act_fun)
        self.actor = build_sequential_network([state_dim, *net_hiddens, action_dim], activation=acts, last_activation=last_act)
        self.critic = build_sequential_network([state_dim, *net_hiddens, 1], acts, None) 
        
        
    def set_action_std(self, new_action_std: float):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            raise ValueError("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
    
    
    def act(self, state: dict):
        env_state = state["env_state"]
        if self.process_image:
            features = self.backbone(state["env_image"].permute(2, 0, 1).unsqueeze(0) / 255)
            env_state = torch.cat((env_state, torch.flatten(features)))

        if self.has_continuous_action_space:
            action_mean = self.actor(env_state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(env_state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(env_state)

        return env_state.detach(), action.detach(), action_logprob.detach(), state_val.detach()
    
    
    def evaluate(self, state: dict, action: torch.Tensor):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy