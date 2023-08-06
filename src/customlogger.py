import torch
import os
from torch.utils.tensorboard import SummaryWriter

class CustomLogger:
    def __init__(self, log_path: str) -> None:
        
        self.path = log_path
        self.writer = SummaryWriter(log_dir=log_path)
        self.file = open(os.path.join(log_path, "logs.csv"), "w+")
        
        self.file.write('episode,timestep,reward\n')
        
        
    def log(self, time_step: int, avg_reward: float, ep: int, loss: torch.Tensor, l_clip: torch.Tensor, \
        l_value_function: torch.Tensor, l_entropy: torch.Tensor, state_values: torch.Tensor, clamped: torch.Tensor):
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Loss : {:.5f}".format(ep, time_step, avg_reward, loss))
        
        self.file.write('{},{},{}\n'.format(ep, time_step, avg_reward))
        self.file.flush()
        
        self.writer.add_scalar('Loss/train', loss, ep)
        self.writer.add_scalar('Clip loss/train', l_clip, ep)
        self.writer.add_scalar('Value function loss/train', l_value_function, ep)
        self.writer.add_scalar('Entropy loss/train', l_entropy, ep)
        self.writer.add_scalar('Reward/train', avg_reward, ep)
        self.writer.add_scalar('State values/train', state_values, ep)
        self.writer.add_scalar('Clamped/train', clamped, ep)
        
        
    def close(self):
        self.file.close()