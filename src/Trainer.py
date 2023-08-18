import torch
import torchaudio
import os
from tqdm import tqdm
import json
import pytorch_warmup as warmup

# from bitsandbytes.optim import AdamW8bit
from contextlib import nullcontext





from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

try:
    import sys
    sys.path.append('src/utils')

    from multi_gpu_helpers import is_main_process
except ModuleNotFoundError:
    from src.utils.multi_gpu_helpers import is_main_process








def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()


    
    
    
    
    
def collate_fn(batch):
    return batch
    
    
class Trainer():
    def __init__(self, model, dataset, dev="cpu", batch_size=32, num_workers=0, prefetch_factor=1, lr=1e-3, save_every_steps=1000, use_scheduler=True, sample_dir="audio_samples", checkpoints_dir="checkpoints", accumulation_steps=1, optimizer_checkpoint=None, scheduler_checkpoint=None, use_amp=True):
        self.model = model
        self.dev = dev
        self.lr = lr
        self.save_every_steps = save_every_steps
        self.use_scheduler = use_scheduler
        self.sample_dir = sample_dir
        self.checkpoints_dir = checkpoints_dir
        self.accumulation_steps = accumulation_steps
        self.optimizer_checkpoint = optimizer_checkpoint
        self.scheduler_checkpoint = scheduler_checkpoint
        self.use_amp = use_amp
        
        
        
        # Put the model on the desired device
        if dev != "cpu":
            # Initialize the environment
            init_distributed()
            
            try:
                local_rank = int(os.environ['LOCAL_RANK'])
            except KeyError:
                local_rank = 0
                print("LOCAL_RANK not found in environment variables. Defaulting to 0.")

            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=False).cuda()
        else:
            self.model = self.model.cpu()
        
        
        
        # Distributed dataloader
        if self.dev == "cpu":
            self.dataloader = DataLoader(dataset, 
                batch_size=batch_size,
                shuffle=True,  
                # collate_fn=lambda x: x,
                collate_fn=collate_fn,
                num_workers=num_workers if num_workers > 0 else 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )
        else:
            self.dataloader = DataLoader(dataset, 
                batch_size=batch_size, 
                # collate_fn=lambda x: x,
                collate_fn=collate_fn,
                num_workers=num_workers if num_workers > 0 else 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                sampler=DistributedSampler(dataset, shuffle=True)
            )
            
            
            
    def train(self, epoch_ckpt=None, step_ckpt=None):
        self.model.train()
        
        if epoch_ckpt is None:
            epoch_ckpt = 0
        if step_ckpt is None:
            step_ckpt = 1
            
        # Model reference is different depending on the device
        if self.dev == "cpu":
            model_ref = self.model
            
            # World size is 1 if not distributed
            world_size = 1
        else:
            model_ref = self.model.module
            
            # Get world size
            world_size = int(os.environ['WORLD_SIZE'])
        
        # Optimzer
        if model_ref.optim_8bit == True:
            optimizer = AdamW8bit(self.model.parameters(), lr=self.lr, eps=1e-7 if self.use_amp else 1e-8) if self.optimizer_checkpoint is None else self.optimizer_checkpoint
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-7 if self.use_amp else 1e-8) if self.optimizer_checkpoint is None else self.optimizer_checkpoint
        
        # Cosine annealing scheduler with warm restarts
        scheduler = None
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6) if self.scheduler_checkpoint is None else self.scheduler_checkpoint
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000*len(self.dataloader), eta_min=1e-6) if self.scheduler_checkpoint is None else self.scheduler_checkpoint
            # https://github.com/Tony-Y/pytorch_warmup
            warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        
        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
        
        num_steps = step_ckpt
        for epoch in range(epoch_ckpt, 10000):
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            if self.dev != "cpu":
                self.dataloader.sampler.set_epoch(epoch)
            
            # Total batch loss
            batch_loss = 0
            
            # Iterate over the batches of data
            for batch_num, batch in enumerate(tqdm(self.dataloader)):
                # Model reference is different depending on the device
                if self.dev == "cpu":
                    model_ref = self.model
                else:
                    model_ref = self.model.module
                
                with torch.no_grad():
                    # Get a batch of data
                    stylized_audio = [x[0] for x in batch]
                    if not model_ref.use_noise:
                        unstylized_audio = [x[1] for x in batch]
                    text = [x[2] for x in batch]
                    conditional_audio = [x[3] if type(x[3]) is not type(None) else torch.zeros(1, 24000) for x in batch]
                    
                    # Max lengths for each type of audio
                    stylized_max_length = max([x.shape[1] for x in stylized_audio])
                    if not model_ref.use_noise:
                        unstylized_max_length = max([x.shape[1] for x in unstylized_audio])
                    conditional_max_length = max([x.shape[1] for x in conditional_audio])
                    
                    # Construct masks for the max length audio. Note that
                    # the stylized masks should be in terms of the unstylized
                    # masks as we won't know how long the stylized audio will
                    # be during inference.
                    # The conditional masks are in terms of themselves
                    # as we know their length during inference.
                    if not model_ref.use_noise:
                        masks_stylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int()#.to(self.device)
                        masks_unstylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int()#.to(self.device)
                    else:
                        masks_stylized = torch.tensor([x.shape[1] for x in stylized_audio]).int()#.to(self.device)
                    masks_conditional = torch.tensor([x.shape[1] for x in conditional_audio]).int()#.to(self.device)
                    
                    # Pad the unstylized audio and conditional audio to the max length
                    if not model_ref.use_noise:
                        unstylized_audio = torch.stack([torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) for a in unstylized_audio])
                    conditional_audio = torch.stack([torch.nn.functional.pad(a, (0, conditional_max_length - a.shape[1], 0, 0)) for a in conditional_audio])
                    
                    # If the stylized audio is shorter than the unstylized audio,
                    # then we need to pad the stylized audio to the same length
                    # as the unstylized audio. If the unstylized audio is shorter
                    # than the stylized audio, then we need to trim the stylized
                    # audio to the same length as the unstylized audio.
                    if not model_ref.use_noise:
                        stylized_audio = torch.stack([
                            torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) 
                                if a.shape[1] < unstylized_max_length 
                                else a[:, :unstylized_max_length] 
                                for a in stylized_audio
                        ])
                    else:
                        stylized_audio = torch.stack([torch.nn.functional.pad(a, (0, stylized_max_length - a.shape[1], 0, 0)) for a in stylized_audio])
                        
                    # The max length of the stylized audio is now the same as the
                    # max length of the unstylized audio. The masks should
                    # also be the same as the unstylized audio.
                    if not model_ref.use_noise:
                        stylized_max_length = unstylized_max_length
                        masks_stylized = masks_unstylized.clone()
                    
                    # # Pad all audio to max lengths. Note that
                    # # we pad the
                    # stylized_audio = torch.stack([torch.nn.functional.pad(a, (0, stylized_max_length - a.shape[1], 0, 0)) for a in stylized_audio])
                    # if not self.use_noise:
                    #     unstylized_audio = torch.stack([torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) for a in unstylized_audio])
                    # conditional_audio = torch.stack([torch.nn.functional.pad(a, (0, conditional_max_length - a.shape[1], 0, 0)) for a in conditional_audio])
                    
                    # Encode the audio using encodec
                    stylized, masks_stylized = model_ref.process_data(stylized_audio, masks_stylized)
                    if not model_ref.use_noise:
                        unstylized, masks_unstylized = model_ref.process_data(unstylized_audio, masks_unstylized)
                    conditional, masks_conditional = model_ref.process_data(conditional_audio, masks_conditional)
                    
                    # if not self.use_noise:
                    #     if stylized.shape[2] < unstylized.shape[2]:
                    #         # Pad the stlyized audio to the same length as the unstylized audio
                    #         stylized = torch.nn.functional.pad(stylized, (0, unstylized.shape[2] - stylized.shape[2], 0, 0))
                    #         masks_stylized = masks_unstylized.clone()
                    #     else:
                    #         # Trim the stylized audio to the same length as the unstylized audio
                    #         stylized = stylized[:, :, :unstylized.shape[2]]
                    #         masks_stylized = masks_unstylized.clone()
                    #         # # Pad the unstlyized audio to the same length as the stylized audio
                    #         # unstylized = torch.nn.functional.pad(unstylized, (0, stylized.shape[2] - unstylized.shape[2], 0, 0))
                    #         # masks_unstylized = masks_stylized.clone()
                    
                    
                    # Audio is of shape (N, E, T)
                    stylized = stylized.clone()#.to(self.device)
                    if not model_ref.use_noise:
                        unstylized = unstylized.clone()#.to(self.device)
                    conditional = conditional.clone()#.to(self.device)
                    # Masks of shape (N, 1, T)
                    masks_stylized = masks_stylized#.to(self.device)
                    if not model_ref.use_noise:
                        masks_unstylized = masks_unstylized#.to(self.device)
                    masks_conditional = masks_conditional#.to(self.device)
                    
                    
                    ## Using the diffusion model utilities, interpolate the audio
                    ## to be some superposition of the stylized and unstylized audio
                    ## Note that instead of predicting the noise as our prior, we 
                    ## predict the unstylized audio as our prior. Basically, the
                    ## unstlyized audio is the "noise" the we want to remove.
                    timesteps = model_ref.diffusion_utils.sample_t(stylized.shape[0])#.to(self.device)
                    positional_embeddings = model_ref.diffusion_utils.t_to_positional_embeddings(timesteps.squeeze(1, -1))#.to(self.device)
                    
                    # Sample the audio as a superposition of the stylized and unstylized audio
                    # or as a uperposition of the stylized audio and noise depending on
                    # what `self.use_noise` is
                    if model_ref.use_noise:
                        unstylized = torch.randn_like(stylized)
                    audio_super = model_ref.diffusion_utils.diffuse_data(timesteps, stylized, unstylized)
                    
                    
                    
                    
                    # Normalize audio by dividing by a scale factor
                    stylized = stylized / model_ref.scale
                    if not model_ref.use_noise:
                        unstylized = unstylized / model_ref.scale
                    audio_super = audio_super / model_ref.scale
                    
                    
                    
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                    # Forward pass to get the predicted unstylized audio
                    # from the interpolated audio
                    model_pred = self.model(audio_super, conditional if conditional_audio is not None else None, positional_embeddings, text, masks_stylized, masks_conditional)
                    
                    
                    
                    # # Tests
                    # with torch.no_grad():
                    #     # audio_super = audio_super * 10_000
                    #     # positional_embeddings = positional_embeddings * 10_000
                    #     comb_sum = masks_stylized[0].sum()
                    #     cond_sum = masks_conditional[0].sum()
                    #     model_pred = self.model.eval()(audio_super.clone(), conditional.clone() if conditional_audio is not None else None, positional_embeddings.clone(), text.copy(), masks_stylized.clone(), masks_conditional.clone())
                    #     # stylized_pred = stylized_pred[:1, :, :comb_sum]
                    #     model_pred *= model_ref.scale
                        
                    #     model_pred_ = self.model.eval()(audio_super[:1, :, :comb_sum], conditional[:1, :, :cond_sum] if conditional_audio is not None else None, positional_embeddings[:1], text[:1].copy())
                    #     model_pred_ *= model_ref.scale
                        
                    #     diff = ((model_pred[0, :, :model_pred_.shape[-1]] - model_pred_)**2).mean()
                    #     max_ = ((model_pred[0, :, :model_pred_.shape[-1]] - model_pred_)**2).max()
                    #     print()
                    
                    
                    # Compute loss depending on the prediction strategy
                    if model_ref.prediction_strategy == "audio":
                        loss = ((stylized-model_pred)**2)
                    elif model_ref.prediction_strategy == "noise":
                        loss = ((unstylized-model_pred)**2)
                    # Mask loss so model isn't biased
                    loss = (loss.flatten(1, -1).sum(1)/(masks_stylized.squeeze(1).sum(1)*stylized.shape[1])).mean() \
                        / self.accumulation_steps
                        
                    # Backward pass
                    if self.use_amp:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Update model every accumulation_steps 
                    if num_steps % self.accumulation_steps == 0:
                        if self.use_amp:
                            grad_scaler.step(optimizer)
                        else:
                            optimizer.step()
                        if self.use_scheduler:
                            with warmup_scheduler.dampening():
                                # scheduler.step()
                                scheduler.step(epoch + batch_num / len(self.dataloader))
                        if self.use_amp:
                            grad_scaler.update()
                        optimizer.zero_grad()
                    
                    if num_steps-step_ckpt < self.save_every_steps and is_main_process():
                        print(f"Step: {num_steps} | Loss: {loss.item()/world_size}")
                    
                    batch_loss += loss.item()
                    num_steps += 1
                
                
                
                
                
                
                
                # Save audio samples
                if num_steps % self.save_every_steps == 0 and is_main_process():
                    with torch.no_grad():
                        print(f"Step: {num_steps} | Loss: {batch_loss / batch_num / world_size}")
                        
                        ## Audio samples
                        if not os.path.exists(f"{self.sample_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.sample_dir}/step_{num_steps}")
                        
                        # Remvoe zero pad from audio
                        if not model_ref.use_noise:
                            unstylized = unstylized[:1, :, :masks_unstylized[0].sum()]
                        else:
                            unstylized = unstylized[:1, :, :masks_stylized[0].sum()]
                        conditional = conditional[:1, :, :masks_conditional[0].sum()]
                            
                        # Generate audio prediction by diffusing the unstylized audio to the predicted stylized audio
                        stylized_audio_pred = model_ref.diffusion_utils.sample_data(model_ref, unstylized, cond=conditional if conditional_audio is not None else None, context=text)
                        stylized_audio_pred *= model_ref.scale
                        # Save audio samples
                        torchaudio.save(f"{self.sample_dir}/step_{num_steps}/stylized.wav", stylized_audio[0].cpu(), 24000)
                        if not model_ref.use_noise:
                            torchaudio.save(f"{self.sample_dir}/step_{num_steps}/unstylized.wav", unstylized_audio[0].cpu(), 24000)
                        torchaudio.save(f"{self.sample_dir}/step_{num_steps}/unstylized_recon.wav", model_ref.encodec_model[0].decoder(stylized_audio_pred)[0].cpu(), 24000)
                        
                        # Save model parameters to json
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        with open(f"{self.checkpoints_dir}/step_{num_steps}/model_params.json", "w") as f:
                            model_ref.defaults["step"] = num_steps+1
                            model_ref.defaults["epoch"] = epoch
                            json.dump(model_ref.defaults, f)
                        
                        # Save model checkpoints
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        torch.save(model_ref.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/model.pth")
                        
                        # Save optimizer checkpoints
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        torch.save(optimizer.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/optimizer.pth")
                        
                        # Save scheduler checkpoints
                        if self.use_scheduler:
                            if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                                os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                            torch.save(scheduler.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/scheduler.pth")
                
                
                
                
                
                
                
                
                
                
                # # Free memory except on last batch
                # if batch_num != len(dataloader)-1:
                del stylized, unstylized, conditional, masks_stylized, masks_conditional, model_pred, audio_super, positional_embeddings, timesteps, text
                if not model_ref.use_noise:
                    del masks_unstylized
                # torch.cuda.empty_cache()
                
            if is_main_process():
                print(f"Epoch: {epoch} | Batch Loss: {batch_loss/len(self.dataloader)/world_size}")