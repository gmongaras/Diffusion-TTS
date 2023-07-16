import torch
from src.utils.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings
from tqdm import tqdm
from copy import deepcopy



# Schedulers from here: https://arxiv.org/pdf/2301.10972.pdf
def linear_scheduler(t, clip_min=1e-9):
    return torch.clamp(1.0 - t, clip_min, 1.0)




class Diffusion_Utils:
    def __init__(self, embedding_dimension):
        self.positional_embeddings = SinusoidalPositionEmbeddings(embedding_dimension)
        
        
        
    # Samples values of t from a uniform distribution between 0 and 1
    # given the batch size
    def sample_t(self, batch_size):
        return torch.rand(batch_size, 1, 1)
    
    
    
    # Transform timesteps into positional embeddings
    # t - batch of timesteps of shape (N, 1, 1)
    # output of shape (N, 1, C)
    def t_to_positional_embeddings(self, t):
        return self.positional_embeddings(t)
    
    
    
    # Given data at timestep 0 and timestep 1, diffusion the data
    # as an interpolation between the two timesteps according to the scheduler
    # t - betach of floating point value between 0 and 1 of shape (N)
    # x_0 - batch of data at timestep 0 (posterior) of shape (N, C, T)
    # x_1 - batch of data at timestep 1 (prior) of shape (N, C, T)
    def diffuse_data(self, t, x_0, x_1):
        # Compute the gamma values for each timestep
        gammas = linear_scheduler(t).reshape(-1, 1, 1)
        
        # Compute the diffusion images
        return torch.sqrt(gammas) * x_0 + torch.sqrt(1 - gammas) * x_1
    
    
    # Given data at timestep t and the predicted origin/prior, x_1,
    # take a DDIM step to get the next timestep
    def take_ddim_step(self, x_t, x_1_pred, t_now, t_next):
        # Compute the gamma values for the timesteps
        gammas = linear_scheduler(t_now).reshape(-1, 1, 1)
        gammas_next = linear_scheduler(t_next).reshape(-1, 1, 1)
        
        # DDIM without noise component
        return torch.sqrt(gammas_next)*((x_t - torch.sqrt(1-gammas)*x_1_pred)/torch.sqrt(gammas)).clamp(-1, 1) + \
            torch.sqrt(1-gammas_next)*x_1_pred
    
    
    # Given a batch of data at timestep 1 (prior), diffuse
    # the data from timestep 1 to timestep 0 (posterior)
    @torch.no_grad()
    def sample_data(self, model, x_1, num_steps=100, cond=None):
        # Iterate over all steps
        x_t = x_1
        for step in tqdm(range(num_steps)):
            # Convert timestep between 0 and 1
            # and start at 1 instead of 0. Then
            # get the next timestep.
            step = torch.tensor(step).to(x_1.device).repeat(x_1.shape[0], 1, 1)
            t = 1-(step/num_steps)
            t_next = (1 - (step+1) / num_steps).clamp(0, 1)
            
            # Predict the original x_1 (prior)
            positional_encodings = self.t_to_positional_embeddings(t.squeeze(1, -1))
            x_1_pred = model(x_t, cond, positional_encodings)
            
            # Take DDPM step on the predicted x_1 prior using DDIM
            x_t = self.take_ddim_step(x_t, x_1_pred, t, t_next)
        
        return x_t