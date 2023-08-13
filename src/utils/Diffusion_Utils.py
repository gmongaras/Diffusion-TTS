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


# Inverse schedulers (given gamma, return t)
def inverse_cosine_schedule(gamma, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = (
        (2/torch.pi) *
        (torch.arccos(
            torch.tensor((
                v_end - (gamma*(v_end-v_start))
            )**(1/(2*tau)))
        )) - start
    ) / (end-start)
    return torch.clamp(output, clip_min, 1.)




class Diffusion_Utils:
    def __init__(self, embedding_dimension, scheduler_type="linear", prediction_strategy="noise"):
        self.positional_embeddings = SinusoidalPositionEmbeddings(embedding_dimension)
        
        if scheduler_type == "linear":
            self.scheduler = linear_schedule
        elif scheduler_type == "sigmoid":
            self.scheduler = sigmoid_schedule
        elif scheduler_type == "cosine":
            self.scheduler = cosine_schedule
        else:
            raise ValueError("Scheduler type not recognized")
        
        self.prediction_strategy = prediction_strategy
        assert prediction_strategy in ["noise", "audio"],\
            "Prediction strategy not recognized"
        
        
        
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
    
    
    
    
    
    # Given a batch of data at timestep 1 (prior), diffuse
    # the data from timestep 1 to timestep 0 (posterior)
    @torch.no_grad()
    def dpm_sample_data(self, model, x_1, num_steps=100, cond=None, context=None, order="first"):
        """
        https://arxiv.org/pdf/2206.00927.pdf
        Our base function is 
            dx_t/dt 
                = h_θ(x_t, t)
                = f(t)x_t + (g^2(t)/2σ_t)ε_θ(x_t, t) where x_T ~ N(0, σ~^2)
                
        If we let
            λ_t = log(α_t/σ_t)
            g(t) 
                = dσ^2/dt - 2dlog(α_t)/dt * σ^2
                = -2σ^2 * dλ_t/dt
            f(t)
                = dlog(α_t)/dt
                
        This can be simplified to
            x_t = α_t/α_s - α_t*∫{from λ_s to λ_t}[e^(-λ) * ε_hat_θ(x_hat_λ, λ) dλ]
            
        Where
            s > 0
            t ∈ [0, s] 
            
            α_t is the mean of our noise distribution at time t
            σ^2_t is the variance of our noise distribution at time t
            
        In case of DDPMs
            α_t = sqrt(γ_t)
            σ^2_t = sqrt(1-γ_t)
            
            where γ_t is our noise scheduler like the cosine scheduler
            
        However, in the paper, they use
            σ_t = sqrt(1-α^2_t)
            
            Which makes it easier for me to derive stuff
        
        This integral basically moves from step s to step t where s > t.
        (note this is order 1)
        
        """
        
        
        # Iterate over all steps
        x_t = x_1
        for step in tqdm(range(1, num_steps)):
            x_t1 = x_t
            
            # Convert timestep between 0 and 1
            # and start at 1 instead of 0. Then
            # get the next timestep.
            step = torch.tensor(step).to(x_1.device).repeat(x_1.shape[0], 1, 1)
            t = 1-((step)/num_steps).clamp(0, 1)          # t is the current step (s)
            t_1 = 1-((step-1)/num_steps).clamp(0, 1)   # t-1 is the previous step
            
            # Compute the gamma values for the timesteps
            # https://arxiv.org/pdf/2301.10972.pdf
            # Note that the gammas here are just the alphas in DDPM
            gammas_t = self.scheduler(t).reshape(-1, 1, 1)
            gammas_t_1 = self.scheduler(t_1).reshape(-1, 1, 1)
            
            # Compute means from the gamma values
            alpha_t = torch.sqrt(gammas_t)
            alpha_t_1 = torch.sqrt(gammas_t_1)
            
            # Compute variances from the gamma values
            # sigma_t = torch.sqrt(1-gammas_t)
            # sigma_t_1 = 1torch.sqrt(-gammas_t_1)
            sigma_t = torch.sqrt(1-alpha_t**2)
            sigma_t_1 = torch.sqrt(1-alpha_t_1**2)
            
            # Lambda values
            lambda_t = torch.log(alpha_t/sigma_t)
            lambda_t1 = torch.log(alpha_t_1/sigma_t_1)
            h = lambda_t - lambda_t1
            
            # First order solver
            if order == "first":
                # Model prediction at t_i-1
                positional_encodings = self.t_to_positional_embeddings(t_1.squeeze(1, -1))
                model_pred = model(x_t1, cond, positional_encodings, context)
                
                # Noise prediction
                if self.prediction_strategy == "audio":
                    noise_pred = (x_t1 - torch.sqrt(gammas_t_1)*model_pred)/torch.sqrt(1-gammas_t_1**2) # Obtained by solving the forward noising equation for x_1
                elif self.prediction_strategy == "noise":
                    noise_pred = model_pred
                
                # Go to next step (t)
                x_t = (alpha_t/alpha_t_1)*x_t1 - sigma_t*(h.exp() - 1)*noise_pred
                print()
            # Second order solver
            elif order == "second":
                # Intermediate timestep
                new_lambda = ((lambda_t1+lambda_t) / 2) # Compute the new, intermedite lambda value
                new_alpha = torch.exp(-0.5*torch.log(torch.exp(-2*new_lambda)+1)) # Alpha vlaue for intermediate step
                new_gamma = new_alpha**2 # Gamma value for intermediate step
                s = inverse_cosine_schedule(new_gamma) # Timestep for intermediate step
                
                # Gamma, Alpha and sigma at s_t
                gammas_s = self.scheduler(s).reshape(-1, 1, 1)
                alpha_s = torch.sqrt(gammas_s)
                sigma_s = torch.sqrt(1-alpha_s**2)
                
                # First model prediction at t-1
                positional_encodings = self.t_to_positional_embeddings(t_1.squeeze(1, -1))
                model_pred = model(x_t1, cond, positional_encodings, context)
                
                # Noise prediction
                if self.prediction_strategy == "audio":
                    noise_pred = (x_t1 - torch.sqrt(gammas_t_1)*model_pred)/torch.sqrt(1-gammas_t_1) # Obtained by solving the forward noising equation for x_1
                elif self.prediction_strategy == "noise":
                    noise_pred = model_pred
                    
                # Move to intermediate step, s
                u = (alpha_s/alpha_t_1)*x_t1 - sigma_s*(torch.exp(h/2) - 1)*noise_pred
                
                # Second model prediction at s
                positional_encodings = self.t_to_positional_embeddings(s.squeeze(1, -1))
                model_pred = model(u, cond, positional_encodings, context)
                
                # Noise prediction
                if self.prediction_strategy == "audio":
                    noise_pred = (u - torch.sqrt(gammas_s)*model_pred)/torch.sqrt(1-gammas_s)
                    # noise_pred = ((alpha_s/alpha_t_1)*x_t1 - u) / (sigma_s*(torch.exp(h/2) - 1))
                elif self.prediction_strategy == "noise":
                    noise_pred = model_pred
                    
                # Go to next step (t)
                x_t = (alpha_t/alpha_t_1)*x_t1 - sigma_t*(torch.exp(h) - 1)*noise_pred
                print()
            else:
                raise ValueError("Order must be first or second")
        
        return x_t 
            
            
        # Iterate over all steps
        x_t = x_1
        for step in tqdm(range(0, num_steps)):
            # Log mean coefficient, alpha
            # alpha_t = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            log_alpha_t = self.scheduler(t).reshape(-1, 1, 1)
            log_alpha_s = self.scheduler(s).reshape(-1, 1, 1)
            
            # Log std coefficient, sigma
            log_sigma_t = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_t))
            log_sigma_s = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_s))
            
            # Lambda t is log(alpha_t) - log(sigma_t)
            lambda_t = log_alpha_t - log_sigma_t
            lambda_s = log_alpha_s - log_sigma_s
            
            h  = lambda_t - lambda_s
            alpha_t = log_alpha_t.exp()
            sigma_t = log_sigma_t.exp()
            sigma_s = log_sigma_s.exp()
            
            phi_1 = torch.expm1(h)
            if model_s is None:
                positional_encodings_s = self.t_to_positional_embeddings(s.squeeze(1, -1))
                model_s = model(x_t, cond, positional_encodings_s, context)
            x_t = (
                sigma_t / sigma_s * x_t
                - alpha_t * phi_1 * model_s
            )
            
            
            
            
            # Take DDIM step on the predicted x_1 prior
            # x_t = self.take_ddim_step(x_t, x_0_pred, t, t_next, torch.all(step==0))
            
            # Take Cold Diffusion step on the predicted x_0 posterior
            # x_t = self.take_cold_diffusion_step(x_t, x_0_pred, t, t_next)
        
        
        
        
        
        
        
        
        ns = self.noise_schedule
        
        order=2
        epsilon=0
        h_init=0.05
        atol=0.0078
        rtol=0.05
        theta=0.9
        t_err=1e-5
        solver_type='dpmsolver'
        
        # s is the starting timestep (1)
        s = torch.ones((1,)).to(x_1)
        zero = torch.ones((1,)).to(x_1)*epsilon
        
        # Alpha values at time s and zero
        # This is the log mean coefficient
        alpha_s = self.scheduler(s).reshape(-1, 1, 1)
        log_alpha_s = torch.log(alpha_s)
        alpha_0 = self.scheduler(zero).reshape(-1, 1, 1)
        log_alpha_0 = torch.log(alpha_0)
        
        # Sigma value at time s and zero
        # This is the log std coefficient
        log_sigma_s = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_s))
        sigma_s = torch.exp(log_sigma_s)
        log_sigma_0 = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_0))
        sigma_0 = torch.exp(log_sigma_0)
        
        # Lambda value at time s and zero
        # Lambda t is log(alpha_t) - log(sigma_t)
        # Lambda is one half of the log SNR
        lambda_s = log_alpha_s - log_sigma_s
        lambda_0 = log_alpha_0 - log_sigma_0
        
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        
        if order == 2:
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            
            r1 = 0.5
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), lambda_0 - lambda_s)
            nfe += order
        print('adaptive solver nfe', nfe)
        return x
        
        
        
        
        
        # Iterate over all steps
        x_t = x_1
        for step in tqdm(range(0, num_steps)):
            # Convert timestep between 0 and 1
            # and start at 1 instead of 0. Then
            # get the next timestep.
            step = torch.tensor(step).to(x_1.device).repeat(x_1.shape[0], 1, 1)
            t = 1-(step/num_steps)
            s = 1-(0/num_steps)
            t_next = (1 - (step+1) / num_steps).clamp(0, 1)
            
            # # Predict the human speech x_0 (posterior)
            # positional_encodings = self.t_to_positional_embeddings(t.squeeze(1, -1))
            # x_0_pred = model(x_t, cond, positional_encodings, context)
            
            # # Compute the gamma values for the timesteps
            # gammas = self.scheduler(t).reshape(-1, 1, 1)
            # gammas_next = self.scheduler(t_next).reshape(-1, 1, 1)
            
            # x_1_pred = (x_t - torch.sqrt(gammas)*x_0_pred)/torch.sqrt(1-gammas) # Obtained by solving the forward noising equation for x_1
            
            # Log mean coefficient, alpha
            # alpha_t = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            log_alpha_t = self.scheduler(t).reshape(-1, 1, 1)
            log_alpha_s = self.scheduler(s).reshape(-1, 1, 1)
            
            # Log std coefficient, sigma
            log_sigma_t = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_t))
            log_sigma_s = 0.5 * torch.log(1 - torch.exp(2 * log_alpha_s))
            
            # Lambda t is log(alpha_t) - log(sigma_t)
            lambda_t = log_alpha_t - log_sigma_t
            lambda_s = log_alpha_s - log_sigma_s
            
            h  = lambda_t - lambda_s
            alpha_t = log_alpha_t.exp()
            sigma_t = log_sigma_t.exp()
            sigma_s = log_sigma_s.exp()
            
            phi_1 = torch.expm1(h)
            if model_s is None:
                positional_encodings_s = self.t_to_positional_embeddings(s.squeeze(1, -1))
                model_s = model(x_t, cond, positional_encodings_s, context)
            x_t = (
                sigma_t / sigma_s * x_t
                - alpha_t * phi_1 * model_s
            )
            
            
            
            
            # Take DDIM step on the predicted x_1 prior
            # x_t = self.take_ddim_step(x_t, x_0_pred, t, t_next, torch.all(step==0))
            
            # Take Cold Diffusion step on the predicted x_0 posterior
            # x_t = self.take_cold_diffusion_step(x_t, x_0_pred, t, t_next)
            
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