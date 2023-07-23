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




class Model():
    def __init__(self, device=torch.device("cpu"), use_noise=False):
        self.device = device
        self.use_noise = use_noise
        self.sampling_rate = 24_000
        self.scale = 10
        
        
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
        embed_dim = 152
        t_embed_dim = 128
        self.model = U_Net(128, 128, embed_dim, 1, num_blocks=2, blk_types=["res", "cond", "res"], cond_dim=128, t_dim=t_embed_dim).to(self.device)
        
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
        
        
        
    
    
    
    def train(self, dataloader):
        # Optimzer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        # Cosine annealing scheduler with warm restarts
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
                    masks_stylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int().to(self.device)
                    if not self.use_noise:
                        masks_unstylized = torch.tensor([x.shape[1] for x in unstylized_audio]).int().to(self.device)
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
                    stylized_audio = torch.stack([
                        torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) 
                            if a.shape[1] < unstylized_max_length 
                            else a[:, :unstylized_max_length] 
                            for a in stylized_audio
                    ])
                        
                    # The max length of the stylized audio is now the same as the
                    # max length of the unstylized audio. The masks should
                    # also be the same as the unstylized audio.
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
                        unstylized = audio_super = torch.randn_like(stylized)
                    else:
                        audio_super = self.diffusion_utils.diffuse_data(timesteps, stylized, unstylized)
                    
                    
                    
                    
                    # Normalize audio by dividing by a scale factor
                    stylized = stylized / self.scale
                    if not self.use_noise:
                        unstylized = unstylized / self.scale
                    audio_super = audio_super / self.scale
                    
                    
                    
                    
                    
                # Forward pass to get the predicted unstylized audio
                # from the interpolated audio
                stylized_pred = self.model(audio_super, conditional if conditional_audio is not None else None, positional_embeddings, masks_stylized, masks_conditional)
                
                # Compute loss
                # We want the model to predict the stylized audio
                loss = loss_fn(stylized, stylized_pred)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step(lambda : epoch + batch_num / len(dataloader))
                scheduler.step()
                
                # print(f"Epoch: {epoch} | Loss: {loss.item()}")
                
                batch_loss += loss.item()
                
                # Free memory except on last batch
                if batch_num != len(dataloader)-1:
                    del stylized, unstylized, conditional, masks_stylized, masks_unstylized, masks_conditional, stylized_pred, audio_super, positional_embeddings, timesteps
                    torch.cuda.empty_cache()
                
            print(f"Epoch: {epoch} | Batch Loss: {batch_loss/len(dataloader)}")
            
            sample_dir = "audio_samples2"
            checkpoints_dir = "checkpoints4"
            
            ## Audio samples
            if not os.path.exists(f"{sample_dir}/epoch_{epoch}"):
                os.makedirs(f"{sample_dir}/epoch_{epoch}")
            
            # Remvoe zero pad from audio
            unstylized = unstylized[:1, :, :masks_unstylized[0].sum()]
            if not self.use_noise:
                conditional = conditional[:1, :, :masks_conditional[0].sum()]
                
            # Generate audio prediction by diffusing the unstylized audio to the predicted stylized audio
            stylized_audio_pred = self.diffusion_utils.sample_data(self.model, unstylized, cond=conditional if conditional_audio is not None else None)
            stylized_audio_pred *= self.scale
            # Save audio samples
            torchaudio.save(f"{sample_dir}/epoch_{epoch}/stylized.wav", stylized_audio[0].cpu(), 24000)
            if not self.use_noise:
                torchaudio.save(f"{sample_dir}/epoch_{epoch}/unstylized.wav", unstylized_audio[0].cpu(), 24000)
            torchaudio.save(f"{sample_dir}/epoch_{epoch}/unstylized_recon.wav", self.encodec_model.decoder(stylized_audio_pred.to(self.device))[0].cpu(), 24000)
            
            # Save model checkpoints
            if not os.path.exists(f"{checkpoints_dir}/epoch_{epoch}"):
                os.makedirs(f"{checkpoints_dir}/epoch_{epoch}")
            torch.save(self.model.state_dict(), f"{checkpoints_dir}/epoch_{epoch}/model.pth")
            
            # Save optimizer checkpoints
            if not os.path.exists(f"{checkpoints_dir}/epoch_{epoch}"):
                os.makedirs(f"{checkpoints_dir}/epoch_{epoch}")
            torch.save(optimizer.state_dict(), f"{checkpoints_dir}/epoch_{epoch}/optimizer.pth")
            
            # Save scheduler checkpoints
            if not os.path.exists(f"{checkpoints_dir}/epoch_{epoch}"):
                os.makedirs(f"{checkpoints_dir}/epoch_{epoch}")
            torch.save(scheduler.state_dict(), f"{checkpoints_dir}/epoch_{epoch}/scheduler.pth")
            
            
            
            
    
    # Given text and a list of conditionals, generate the stylized audio
    @torch.no_grad()
    def infer(self, text, conditionals=[], num_steps=100):
        # Create the unstylized audio
        try:
            unstylized = torch.tensor(self.tts.tts(text)).float()
        except RuntimeError:
            unstylized = torch.tensor(self.tts.tts(text + "...")).float()
        
        # Load in the conditional audio
        conditionals = [torchaudio.load(path) for path in conditionals]
        conditionals = [torchaudio.transforms.Resample(c[1], 24000)(c[0]) for c in conditionals]
        
        # Pass the data through the encodec
        unstylized, _ = self.process_data(unstylized.unsqueeze(0))
        conditionals = [self.process_data(c)[0] for c in conditionals]
        
        # Pad the unstylized audio to the nearest second
        # Note: 75 is a second in the latent dimension
        unstylized = torch.nn.functional.pad(unstylized, (0, 75 - unstylized.shape[2]%75, 0, 0))
        
        # Concatenate the conditional audio along the time dimension
        conditionals = torch.cat(conditionals, dim=2)
        
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
        self.model.load_state_dict(torch.load(path + "/model.pth", map_location=self.device))
        self.model.eval()
        
        # Load in the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        optimizer.load_state_dict(torch.load(path + "/optimizer.pth", map_location=self.device))
        
        return optimizer
