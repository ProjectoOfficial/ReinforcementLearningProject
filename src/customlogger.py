import os
from torch.utils.tensorboard import SummaryWriter

class CustomLogger:
    def __init__(self, log_path) -> None:
        
        self.path = log_path
        self.writer = SummaryWriter(log_dir=log_path)
        self.file = open(os.path.join(log_path, "logs.csv"), "w+")
        
        self.file.write('episode,timestep,reward\n')
        
        
    def log(self, time_step, avg_reward, i_episode, loss, l_clip, l_value_function, l_entropy, state_values, clamped):
        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {} \t\t Loss : {:.5f}".format(i_episode, time_step, avg_reward, loss))
        
        self.file.write('{},{},{}\n'.format(i_episode, time_step, avg_reward))
        self.file.flush()
        
        self.writer.add_scalar('Loss/train', loss, i_episode)
        self.writer.add_scalar('Clip loss/train', l_clip, i_episode)
        self.writer.add_scalar('Value function loss/train', l_value_function, i_episode)
        self.writer.add_scalar('Entropy loss/train', l_entropy, i_episode)
        self.writer.add_scalar('Reward/train', avg_reward, i_episode)
        self.writer.add_scalar('State values/train', state_values, i_episode)
        self.writer.add_scalar('Clamped/train', clamped, i_episode)
        
        
    def close(self):
        self.file.close()