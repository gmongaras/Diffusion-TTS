import torch
import torchaudio
import torch.nn as nn
import numpy as np
import os

from transformers import EncodecModel, AutoProcessor
from TTS.api import TTS
from src.models.Transformer import Transformer
from src.models.U_Net import U_Net




class Model():
    def __init__(self, ):
        
        
        ### Encodec
        # load the model + processor (for pre-processing the audio)
        self.encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().cuda()
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        # Get the quantizer from the model
        self.quantizer = self.encodec_model.quantizer
        
        
        
        ### TTS model
        model_name = 'tts_models/en/ljspeech/speedy-speech'
        self.tts = TTS(model_name, gpu=False)
        
        
        # Transformer model to train
        # self.model = Transformer(128, 128, 512, 8).cuda()
        self.model = U_Net(128, 128, 128, 1, num_blocks=2, blk_types=["res", "cond", "res"], cond_dim=128).cuda()
        
        # Paramater counts
        print("U-Net model has {} parameters".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        
        
        
        
        
    def process_data(self, audio_segment):
        # Preporcess the audio segments
        audio_segment = self.processor(audio_segment.squeeze(1).tolist(), sampling_rate=24000, return_tensors="pt")
        
        
        # Encode inputs
        encoded_outputs = self.encodec_model.encode(audio_segment["input_values"].cuda(), audio_segment["padding_mask"].cuda(), bandwidth=24.0)
        encoded_outputs = encoded_outputs.audio_codes
        
        
        # Dequantize the outputs.
        # Note the quantizer expects inputs to be of shape (CB, B, T)
        encoded_outputs = self.quantizer.decode(encoded_outputs.squeeze(0).transpose(0, 1))
        
        # Transpose the inputs to (B, T, E)
        encoded_outputs = encoded_outputs.transpose(1, 2).float().cuda()
        
        return encoded_outputs
        
        
        
    
    
    
    def train(self, dataloader):
        # Optimzer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        
        
        
        
        
        
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
            batch_loss_1 = 0
            
            # Iterate over the batches of data
            for batch in dataloader:
                with torch.no_grad():
                    # Get a batch of data
                    stylized_audio = [x[0] for x in batch]
                    unstylized_audio = [x[1] for x in batch]
                    text = [x[2] for x in batch]
                    conditional_audio = [x[3] if type(x[3]) is not type(None) else torch.zeros(1, 16000) for x in batch]
                    
                    # Max lengths for each type of audio
                    stylized_max_length = max([x.shape[1] for x in stylized_audio])
                    unstylized_max_length = max([x.shape[1] for x in unstylized_audio])
                    conditional_max_length = max([x.shape[1] for x in conditional_audio])
                    
                    # Pad all audio to max lengths
                    stylized_audio = torch.stack([torch.nn.functional.pad(a, (0, stylized_max_length - a.shape[1], 0, 0)) for a in stylized_audio])
                    unstylized_audio = torch.stack([torch.nn.functional.pad(a, (0, unstylized_max_length - a.shape[1], 0, 0)) for a in unstylized_audio])
                    conditional_audio = torch.stack([torch.nn.functional.pad(a, (0, conditional_max_length - a.shape[1], 0, 0)) for a in conditional_audio])
                    
                    # Encode the audio using encodec
                    encoder_outputs_stylized = self.process_data(stylized_audio)
                    encoder_outputs_unstylized = self.process_data(unstylized_audio)
                    encoder_outputs_conditional = self.process_data(unstylized_audio)
                    
                    if encoder_outputs_stylized.shape[1] < encoder_outputs_unstylized.shape[1]:
                        # Pad the stlyized audio to the same length as the unstylized audio
                        padding_pre = (encoder_outputs_unstylized.shape[1] - encoder_outputs_stylized.shape[1])//2
                        padding_post = (encoder_outputs_unstylized.shape[1] - encoder_outputs_stylized.shape[1]) - padding_pre
                        encoder_outputs_stylized = torch.nn.functional.pad(encoder_outputs_stylized, (0, 0, padding_pre, padding_post))
                    else:
                        # Pad the unstlyized audio to the same length as the stylized audio
                        padding_pre = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1])//2
                        padding_post = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1]) - padding_pre
                        encoder_outputs_unstylized = torch.nn.functional.pad(encoder_outputs_unstylized, (0, 0, padding_pre, padding_post))
                    
                # Forward pass
                unstylized_audio_recon = self.model(encoder_outputs_unstylized.permute(0, 2, 1).clone().cuda(), encoder_outputs_conditional.permute(0, 2, 1).clone().cuda() if conditional_audio is not None else None)
                
                # Compute loss
                loss = loss_fn(encoder_outputs_stylized, unstylized_audio_recon.permute(0, 2, 1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print(f"Epoch: {epoch} | Loss: {loss.item()}")
                
                batch_loss_1 += loss.item()
                
            print(f"Epoch: {epoch} | Batch Loss: {batch_loss_1/len(dataloader)}")
            
            # Audio samples
            if not os.path.exists("audio_samples/epoch_{}".format(epoch)):
                os.makedirs("audio_samples/epoch_{}".format(epoch))
            torchaudio.save(f"audio_samples/epoch_{epoch}/stylized.wav", stylized_audio[0].cpu(), 24000)
            torchaudio.save(f"audio_samples/epoch_{epoch}/unstylized.wav", unstylized_audio[0].cpu(), 24000)
            torchaudio.save(f"audio_samples/epoch_{epoch}/unstylized_recon.wav", self.encodec_model.decoder(unstylized_audio_recon)[0].cpu(), 24000)
            
            # Save model checkpoints
            if not os.path.exists("checkpoints/epoch_{}".format(epoch)):
                os.makedirs("checkpoints/epoch_{}".format(epoch))
            torch.save(self.model.state_dict(), f"checkpoints/epoch_{epoch}/model.pth")
            
            # Save optimizer checkpoints
            if not os.path.exists("checkpoints/epoch_{}".format(epoch)):
                os.makedirs("checkpoints/epoch_{}".format(epoch))
            torch.save(optimizer.state_dict(), f"checkpoints/epoch_{epoch}/optimizer.pth")
