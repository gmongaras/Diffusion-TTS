import torch
import torchaudio
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

from transformers import EncodecModel, AutoProcessor
from TTS.api import TTS
from src.models.Transformer import Transformer
from src.models.U_Net import U_Net
from src.utils.Diffusion_Utils import Diffusion_Utils




class Model(nn.Module):
    def __init__(self, embed_dim=128, t_embed_dim=128, cond_embed_dim=128, num_blocks=2, blk_types=["res", "cond2", "res"], device=torch.device("cpu"), use_noise=False, use_scheduler=True):
        super(Model, self).__init__()
        
        self.device = device
        self.use_noise = use_noise
        self.sampling_rate = 24_000
        self.scale = 10
        self.use_scheduler = False
        
        self.embed_dim = embed_dim
        self.t_embed_dim = t_embed_dim
        
        
        ### Encodec
        # load the model + processor (for pre-processing the audio)
        self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        # Get the quantizer from the model
        self.quantizer = self.encodec_model.quantizer
        
        
        
        ### TTS model
        model_name = 'tts_models/en/ljspeech/speedy-speech'
        self.tts = TTS(model_name, gpu=False)
        
        
        # Model to train
        # self.model = Transformer(128, 128, 512, 8).to(self.device)
        self.model = U_Net(128, 128, embed_dim, 1, num_blocks=num_blocks, blk_types=blk_types, cond_dim=cond_embed_dim, t_dim=t_embed_dim).to(self.device)
        
        # Diffusion model utility class
        self.diffusion_utils = Diffusion_Utils(t_embed_dim)
        
        # Paramater counts
        print("U-Net model has {} parameters".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        
        
        
        
    def process_data(self, audio_segment, masks=None):
        # Preporcess the audio segments
        audio_segment = self.processor(audio_segment.squeeze(1).tolist(), sampling_rate=24000, return_tensors="pt")
        
        # Change the masks
        if masks is not None:
            for i, m in enumerate(masks):
                audio_segment["padding_mask"][i][m:] = 0
        
        # Encode inputs
        # NOTE: Encodec is stupid and expects the masks to be 0 where we want to
        # keep the audio and 1 where we want to mask it out
        encoded_outputs = self.encodec_model.encode(audio_segment["input_values"].to(self.device), audio_segment["padding_mask"].to(self.device), bandwidth=24.0)
        encoded_outputs = encoded_outputs.audio_codes
        
        
        # Dequantize the outputs.
        # Note the quantizer expects inputs to be of shape (CB, B, T)
        encoded_outputs = self.quantizer.decode(encoded_outputs.squeeze(0).transpose(0, 1))
        
        # Transpose the inputs to (B, E, T)
        encoded_outputs = encoded_outputs.float().to(self.device)
        
        # Change the masks to be in the context of the
        # latent dimension and make it a full matrix.
        # This matrix will be zero where we mask
        # and 1 where we keep
        if masks is not None:
            masks_full = torch.ones(encoded_outputs.shape[0], 1, encoded_outputs.shape[2], dtype=torch.bool, device=self.device)
            for i, m in enumerate(masks):
                # New mask is basically quantized from 24_000 -> 75
                new_mask = m//(self.sampling_rate//self.quantizer.frame_rate)
                masks_full[i][:, new_mask:] = 0
                
        # Return masked outputs
        if masks is None:
            return encoded_outputs, None
        return encoded_outputs*masks_full, masks_full
        
        
        
    
    
    
    def train_model(self, dataloader, lr=1e-3, save_every_steps=1000, sample_dir="audio_samples", checkpoints_dir="checkpoints", accumulation_steps=1, optimizer_checkpoint=None):
        self.train()
        
        # Optimzer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr) if optimizer_checkpoint is None else optimizer_checkpoint
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        # Cosine annealing scheduler with warm restarts
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        
        
        
        
        
        
        
        # stylized_audio, stylized_sr = torchaudio.load("notebooks/sample.wav")

        # # Text for this audio
        # text = "Mr. Quilter is the Apostle of the Middle Class and we are glad to welcome his gospel."
        
        # # Generate an unstylized audio clip based off the same text
        # unstylized_audio = torch.tensor(self.tts.tts(text)).float()
        # sr = 22050
        
        # # Batch the audio
        # stylized_audio = stylized_audio.repeat(2, 1, 1)
        # unstylized_audio = unstylized_audio.repeat(2, 1, 1)
        
        # # Resample both audio clips to 24khz
        # stylized_audio = torchaudio.transforms.Resample(stylized_sr, 24000)(stylized_audio).squeeze()
        # unstylized_audio = torchaudio.transforms.Resample(sr, 24000)(unstylized_audio).squeeze()
        

        
        
        # encoder_outputs_stylized = self.process_data(stylized_audio)
        # encoder_outputs_unstylized = self.process_data(unstylized_audio)
        
        
        
        
        
        
        
        num_steps = 0
        for epoch in range(0, 10000):
            # Total batch loss
            batch_loss = 0
            
            # Iterate over the batches of data
            for batch_num, batch in enumerate(tqdm(dataloader)):
                with torch.no_grad():
                    # Get a batch of data
                    stylized_audio = [x[0] for x in batch]
                    if not self.use_noise:
                        unstylized_audio = [x[1] for x in batch]
                    text = [x[2] for x in batch]
                    conditional_audio = [x[3] if type(x[3]) is not type(None) else torch.zeros(1, 24000) for x in batch]
                    
                    # Max lengths for each type of audio
                    stylized_max_length = max([x.shape[1] for x in stylized_audio])
                    if not self.use_noise:
                        unstylized_max_length = max([x.shape[1] for x in unstylized_audio])
                    conditional_max_length = max([x.shape[1] for x in conditional_audio])
                    
                    # Construct masks for the max length audio. Note that
                    # the stylized masks should be in terms of the unstylized
                    # masks as we won't know how long the stylized audio will
                    # be during inference.
                    # The conditional masks are in terms of themselves
                    # as we know their length during inference.
                    if not self.use_noise:
                        masks_stylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int().to(self.device)
                        masks_unstylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int().to(self.device)
                    else:
                        masks_stylized = torch.tensor([x.shape[1] for x in stylized_audio]).int().to(self.device)
                    masks_conditional = torch.tensor([x.shape[1] for x in conditional_audio]).int().to(self.device)
                    
                    # Pad the unstylized audio and conditional audio to the max length
                    if not self.use_noise:
                        unstylized_audio = torch.stack([torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) for a in unstylized_audio])
                    conditional_audio = torch.stack([torch.nn.functional.pad(a, (0, conditional_max_length - a.shape[1], 0, 0)) for a in conditional_audio])
                    
                    # If the stylized audio is shorter than the unstylized audio,
                    # then we need to pad the stylized audio to the same length
                    # as the unstylized audio. If the unstylized audio is shorter
                    # than the stylized audio, then we need to trim the stylized
                    # audio to the same length as the unstylized audio.
                    if not self.use_noise:
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
                    if not self.use_noise:
                        stylized_max_length = unstylized_max_length
                        masks_stylized = masks_unstylized.clone()
                    
                    # # Pad all audio to max lengths. Note that
                    # # we pad the
                    # stylized_audio = torch.stack([torch.nn.functional.pad(a, (0, stylized_max_length - a.shape[1], 0, 0)) for a in stylized_audio])
                    # if not self.use_noise:
                    #     unstylized_audio = torch.stack([torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) for a in unstylized_audio])
                    # conditional_audio = torch.stack([torch.nn.functional.pad(a, (0, conditional_max_length - a.shape[1], 0, 0)) for a in conditional_audio])
                    
                    # Encode the audio using encodec
                    stylized, masks_stylized = self.process_data(stylized_audio, masks_stylized)
                    if not self.use_noise:
                        unstylized, masks_unstylized = self.process_data(unstylized_audio, masks_unstylized)
                    conditional, masks_conditional = self.process_data(conditional_audio, masks_conditional)
                    
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
                    stylized = stylized.clone().to(self.device)
                    if not self.use_noise:
                        unstylized = unstylized.clone().to(self.device)
                    conditional = conditional.clone().to(self.device)
                    # Masks of shape (N, 1, T)
                    masks_stylized = masks_stylized.to(self.device)
                    if not self.use_noise:
                        masks_unstylized = masks_unstylized.to(self.device)
                    masks_conditional = masks_conditional.to(self.device)
                    
                    
                    ## Using the diffusion model utilities, interpolate the audio
                    ## to be some superposition of the stylized and unstylized audio
                    ## Note that instead of predicting the noise as our prior, we 
                    ## predict the unstylized audio as our prior. Basically, the
                    ## unstlyized audio is the "noise" the we want to remove.
                    timesteps = self.diffusion_utils.sample_t(stylized.shape[0]).to(self.device)
                    positional_embeddings = self.diffusion_utils.t_to_positional_embeddings(timesteps.squeeze(1, -1)).to(self.device)
                    
                    # Sample the audio as a superposition of the stylized and unstylized audio
                    # or as a uperposition of the stylized audio and noise depending on
                    # what `self.use_noise` is
                    if self.use_noise:
                        unstylized = torch.randn_like(stylized)
                    audio_super = self.diffusion_utils.diffuse_data(timesteps, stylized, unstylized)
                    
                    
                    
                    
                    # Normalize audio by dividing by a scale factor
                    stylized = stylized / self.scale
                    if not self.use_noise:
                        unstylized = unstylized / self.scale
                    audio_super = audio_super / self.scale
                    
                    
                    
                    
                    
                # Forward pass to get the predicted unstylized audio
                # from the interpolated audio
                stylized_pred = self.model(audio_super, conditional if conditional_audio is not None else None, positional_embeddings, masks_stylized, masks_conditional)
                
                
                
                # # Tests
                # with torch.no_grad():
                #     # audio_super = audio_super * 10_000
                #     # positional_embeddings = positional_embeddings * 10_000
                #     comb_sum = masks_stylized[0].sum()
                #     cond_sum = masks_conditional[0].sum()
                #     stylized_pred = self.model.eval()(audio_super.clone(), conditional.clone() if conditional_audio is not None else None, positional_embeddings.clone(), masks_stylized.clone(), masks_conditional.clone())
                #     # stylized_pred = stylized_pred[:1, :, :comb_sum]
                #     stylized_pred *= self.scale
                    
                #     stylized_pred_ = self.model.eval()(audio_super[:1, :, :comb_sum], conditional[:1, :, :cond_sum] if conditional_audio is not None else None, positional_embeddings[:1])
                #     stylized_pred_ *= self.scale
                    
                #     diff = ((stylized_pred[0, :, :stylized_pred_.shape[-1]] - stylized_pred_)**2).mean()
                #     max_ = ((stylized_pred[0, :, :stylized_pred_.shape[-1]] - stylized_pred_)**2).max()
                #     print()
                
                
                # Compute loss
                # We want the model to predict the stylized audio
                # Note that we also need to account for the masked terms so
                # the average doesn't get thrown off. With the mask, the
                # average will be a lot lower for shorter sequences.
                # loss = loss_fn(stylized, stylized_pred)
                loss = (((stylized-stylized_pred)**2).flatten(1, -1).sum(1)/(masks_stylized.squeeze(1).sum(1)*stylized.shape[1])).mean()
                
                # Backward pass
                loss.backward()
                   
                # Update model every accumulation_steps 
                if num_steps % accumulation_steps == 0:
                    optimizer.step()
                    if self.use_scheduler:
                        scheduler.step(lambda : epoch + batch_num / len(dataloader))
                    optimizer.zero_grad()
                
                if epoch == 0:
                    print(f"Step: {num_steps} | Loss: {loss.item()}")
                
                batch_loss += loss.item()
                num_steps += 1
                
                
                
                
                
                
                
                # Save audio samples
                if num_steps % save_every_steps == 0:
                    with torch.no_grad():
                        print(f"Step: {num_steps} | Loss: {batch_loss / num_steps}")
                        
                        ## Audio samples
                        if not os.path.exists(f"{sample_dir}/step_{num_steps}"):
                            os.makedirs(f"{sample_dir}/step_{num_steps}")
                        
                        # Remvoe zero pad from audio
                        if not self.use_noise:
                            unstylized = unstylized[:1, :, :masks_unstylized[0].sum()]
                        else:
                            unstylized = unstylized[:1, :, :masks_stylized[0].sum()]
                        conditional = conditional[:1, :, :masks_conditional[0].sum()]
                            
                        # Generate audio prediction by diffusing the unstylized audio to the predicted stylized audio
                        stylized_audio_pred = self.diffusion_utils.sample_data(self.model, unstylized, cond=conditional if conditional_audio is not None else None)
                        stylized_audio_pred *= self.scale
                        # Save audio samples
                        torchaudio.save(f"{sample_dir}/step_{num_steps}/stylized.wav", stylized_audio[0].cpu(), 24000)
                        if not self.use_noise:
                            torchaudio.save(f"{sample_dir}/step_{num_steps}/unstylized.wav", unstylized_audio[0].cpu(), 24000)
                        torchaudio.save(f"{sample_dir}/step_{num_steps}/unstylized_recon.wav", self.encodec_model.decoder(stylized_audio_pred.to(self.device))[0].cpu(), 24000)
                        
                        # Save model checkpoints
                        if not os.path.exists(f"{checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{checkpoints_dir}/step_{num_steps}")
                        torch.save(self.state_dict(), f"{checkpoints_dir}/step_{num_steps}/model.pth")
                        
                        # Save optimizer checkpoints
                        if not os.path.exists(f"{checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{checkpoints_dir}/step_{num_steps}")
                        torch.save(optimizer.state_dict(), f"{checkpoints_dir}/step_{num_steps}/optimizer.pth")
                        
                        # Save scheduler checkpoints
                        if self.use_scheduler:
                            if not os.path.exists(f"{checkpoints_dir}/step_{num_steps}"):
                                os.makedirs(f"{checkpoints_dir}/step_{num_steps}")
                            torch.save(scheduler.state_dict(), f"{checkpoints_dir}/step_{num_steps}/scheduler.pth")
                
                
                
                
                
                
                
                
                
                
                # # Free memory except on last batch
                # if batch_num != len(dataloader)-1:
                del stylized, unstylized, conditional, masks_stylized, masks_conditional, stylized_pred, audio_super, positional_embeddings, timesteps
                if not self.use_noise:
                    del masks_unstylized
                torch.cuda.empty_cache()
                
            print(f"Epoch: {epoch} | Batch Loss: {batch_loss/len(dataloader)}")
            
            
            
            
    
    # Given text and a list of conditionals, generate the stylized audio
    @torch.no_grad()
    def infer(self, text, conditionals=[], num_steps=100):
        # Create the unstylized audio
        try:
            unstylized = torch.tensor(self.tts.tts(text)).float()
        except RuntimeError:
            unstylized = torch.tensor(self.tts.tts(text + "...")).float()
            
        # Resample generated audio to 24000 Hz
        unstylized = torchaudio.transforms.Resample(22050, 24000)(unstylized)
        
        # Load in the conditional audio
        conditionals = [torchaudio.load(path) for path in conditionals]
        conditionals = torch.cat([torchaudio.transforms.Resample(c[1], 24000)(c[0]) for c in conditionals], dim=1)
        
        # Pass the data through the encodec
        unstylized, _ = self.process_data(unstylized.unsqueeze(0))
        # conditionals = [self.process_data(c)[0] for c in conditionals]
        conditionals, _ = self.process_data(conditionals)
        
        # Pad the unstylized audio to the nearest second
        # Note: 75 is a second in the latent dimension
        # unstylized = torch.nn.functional.pad(unstylized, (0, 75 - unstylized.shape[2]%75, 0, 0))
        
        # Concatenate the conditional audio along the time dimension
        # conditionals = torch.cat(conditionals, dim=2)
        
        # Permute audio to be of shape (N, E, T)
        unstylized = unstylized.to(self.device) / self.scale
        conditionals = conditionals.to(self.device) / self.scale
        
        # Get prediction
        pred = self.diffusion_utils.sample_data(self.model, unstylized, num_steps=num_steps, cond=conditionals)
        pred *= self.scale
        
        # Decode the audio
        return self.encodec_model.decoder(pred.to(self.device))[0].cpu()
    
    
    
    
    
    # Used to load in checkpoints
    def load_checkpoint(self, path):
        # Load in the model
        self.load_state_dict(torch.load(path + "/model.pth", map_location=self.device))
        self.eval()
        
        # Load in the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        optimizer.load_state_dict(torch.load(path + "/optimizer.pth", map_location=self.device))
        
        return optimizer
