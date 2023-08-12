import torch
from tqdm import tqdm
from copy import deepcopy
import math

try: # For distributed training
    from src.utils.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings
except ModuleNotFoundError:
    from utils.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings



# Schedulers from here: https://arxiv.org/pdf/2301.10972.pdf
def linear_schedule(t, clip_min=1e-9):
    return torch.clamp(1.0 - t, clip_min, 1.0)
def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    # A gamma function based on sigmoid function.
    v_start = torch.sigmoid(start / tau)
    v_end = torch.sigmoid(end / tau)
    output = torch.sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clamp(output, clip_min, 1.)
def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clamp(output, clip_min, 1.)




class Diffusion_Utils:
    def __init__(self, embedding_dimension, scheduler_type="linear"):
        self.positional_embeddings = SinusoidalPositionEmbeddings(embedding_dimension)
        
        if scheduler_type == "linear":
            self.scheduler = linear_schedule
        elif scheduler_type == "sigmoid":
            self.scheduler = sigmoid_schedule
        elif scheduler_type == "cosine":
            self.scheduler = cosine_schedule
        else:
            raise ValueError("Scheduler type not recognized")
        
        
        
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
    # NOTE: The forward noising equation is x_t = sqrt(gammas)*x_0 + sqrt(1-gammas)*x_1
    def diffuse_data(self, t, x_0, x_1):
        # Compute the gamma values for each timestep
        gammas = self.scheduler(t).reshape(-1, 1, 1).to(x_0.device)
        
        # Compute the diffusion images
        return torch.sqrt(gammas) * x_0 + torch.sqrt(1 - gammas) * x_1
    
    
    # Given data at timestep t and the predicted origin/prior, x_1,
    # take a DDIM step to get the next timestep
    def take_ddim_step(self, x_t, x_0_pred, t_now, t_next, clamp=False):
        # Compute the gamma values for the timesteps
        gammas = self.scheduler(t_now).reshape(-1, 1, 1)
        gammas_next = self.scheduler(t_next).reshape(-1, 1, 1)
        
        x_1_pred = (x_t - torch.sqrt(gammas)*x_0_pred)/torch.sqrt(1-gammas) # Obtained by solving the forward noising equation for x_1
        
        # DDIM without noise component
        if clamp:
            return torch.sqrt(gammas_next)*((x_t - torch.sqrt(1-gammas)*x_1_pred)/torch.sqrt(gammas)).clamp(-25, 25) + \
                torch.sqrt(1-gammas_next)*x_1_pred
        return torch.sqrt(gammas_next)*((x_t - torch.sqrt(1-gammas)*x_1_pred)/torch.sqrt(gammas)) + \
            torch.sqrt(1-gammas_next)*x_1_pred
            
            
    def take_cold_diffusion_step(self, x_t, x_0_pred, t_now, t_next):
        # Compute the gamma values for the timesteps
        gammas = self.scheduler(t_now).reshape(-1, 1, 1)
        gammas_next = self.scheduler(t_next).reshape(-1, 1, 1)
        
        # Get the prediction for x_0 (posterior) and x_1 (prior)
        x_0_hat = x_0_pred
        x_1_hat = (x_t-torch.sqrt(gammas)*x_0_hat)/torch.sqrt(1-gammas) # Obtained by solving the forward noising equation for x_1
        
        # Predicted x_t
        xt_bar = self.diffuse_data(t_now, x_0_hat, x_1_hat)
        #xt_bar = gammas*x1_bar + (1-gammas)*x2_bar
        
        # Predicted x_t-1
        xt_sub1_bar = self.diffuse_data(t_next, x_0_hat, x_1_hat)
        #xt_sub1_bar = gammas_next*x1_bar + (1-gammas_next)*x2_bar
        
        # Get corrected prediction for x_t-1
        # NOTE: x_t - xt_bar predicts x_0
        #       and xt_sub1_bar predicts x_t-1 from the predicted x_0 
        return x_t - xt_bar + xt_sub1_bar
    
    
    # Given a batch of data at timestep 1 (prior), diffuse
    # the data from timestep 1 to timestep 0 (posterior)
    @torch.no_grad()
    def sample_data(self, model, x_1, num_steps=100, cond=None, context=None):
        # Iterate over all steps
        x_t = x_1
        for step in tqdm(range(0, num_steps)):
            # Convert timestep between 0 and 1
            # and start at 1 instead of 0. Then
            # get the next timestep.
            step = torch.tensor(step).to(x_1.device).repeat(x_1.shape[0], 1, 1)
            t = 1-(step/num_steps)
            t_next = (1 - (step+1) / num_steps).clamp(0, 1)
            
            # Predict the human speech x_0 (posterior)
            positional_encodings = self.t_to_positional_embeddings(t.squeeze(1, -1))
            x_0_pred = model(x_t/1, cond, positional_encodings, context)*1
            
            # Take DDIM step on the predicted x_1 prior
            # x_t = self.take_ddim_step(x_t, x_0_pred, t, t_next, torch.all(step==0))
            
            # Take Cold Diffusion step on the predicted x_0 posterior
            x_t = self.take_cold_diffusion_step(x_t, x_0_pred, t, t_next)
            
        # ### Final prediction
        # step = num_steps
        
        # # Convert timestep between 0 and 1
        # # and start at 1 instead of 0. Then
        # # get the next timestep.
        # step = torch.tensor(step).to(x_1.device).repeat(x_1.shape[0], 1, 1)
        # t = 1-(step/num_steps)
        
        # # Predict the original x_1 (prior)
        # positional_encodings = self.t_to_positional_embeddings(t.squeeze(1, -1))
        # x_0_pred = model(x_t, cond, positional_encodings)
        
        return x_0_pred