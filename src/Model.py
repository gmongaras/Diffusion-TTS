import torch
import torchaudio
import torch.nn as nn
import numpy as np

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
        self.model = Transformer(128, 128, 128, 4).cuda()
        self.model2 = U_Net(128, 128, 128, 1, 128, num_blocks=2, blk_types=["res", "res"]).cuda()
        
        
        
        
        
    def process_data(self, audio_segment):
        # Preporcess the audio segments
        audio_segment = self.processor(audio_segment.tolist(), sampling_rate=24000, return_tensors="pt")
        
        
        # Encode inputs
        encoude_outputs = self.encodec_model.encode(audio_segment["input_values"].cuda(), audio_segment["padding_mask"].cuda(), bandwidth=24.0)
        encoude_outputs = encoude_outputs.audio_codes
        
        
        # Dequantize the outputs.
        # Note the quantizer expects inputs to be of shape (CB, B, T)
        encoude_outputs = self.quantizer.decode(encoude_outputs.squeeze().transpose(0, 1))
        
        # Transpose the inputs to (B, T, E)
        encoude_outputs = encoude_outputs.squeeze().transpose(1, 2).float().cuda()
        
        return encoude_outputs
        
        
        
    
    
    
    def train(self, dataloader):
        # Optimzer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        optimizer2 = torch.optim.AdamW(self.model2.parameters(), lr=5e-4)
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        
        
        
        
        
        
        stylized_audio, stylized_sr = torchaudio.load("notebooks/sample.wav")

        # Text for this audio
        text = "Mr. Quilter is the Apostle of the Middle Class and we are glad to welcome his gospel."
        
        # Generate an unstylized audio clip based off the same text
        unstylized_audio = torch.tensor(self.tts.tts(text)).float()
        sr = 22050
        
        # Batch the audio
        stylized_audio = stylized_audio.repeat(2, 1, 1)
        unstylized_audio = unstylized_audio.repeat(2, 1, 1)
        
        # Resample both audio clips to 24khz
        stylized_audio = torchaudio.transforms.Resample(stylized_sr, 24000)(stylized_audio).squeeze()
        unstylized_audio = torchaudio.transforms.Resample(sr, 24000)(unstylized_audio).squeeze()
        

        
        
        encoder_outputs_stylized = self.process_data(stylized_audio)
        encoder_outputs_unstylized = self.process_data(unstylized_audio)
        
        
        
        
        
        
        
        for epoch in range(0, 10000):
            # Get a batch of data
            # stylized_audio, unstylized_audio, text = next(iter(dataloader))
            
            # encoder_outputs_stylized = self.process_data(stylized_audio)
            # encoder_outputs_unstylized = self.process_data(unstylized_audio)
            
            # Pad the unstlyized audio to the same length as the stylized audio
            padding_pre = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1])//2
            padding_post = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1]) - padding_pre
            encoder_outputs_unstylized = torch.nn.functional.pad(encoder_outputs_unstylized, (0, 0, padding_pre, padding_post))
            
            # Forward pass
            unstylized_audio_recon = self.model(encoder_outputs_unstylized.clone().cuda(), encoder_outputs_stylized.clone().cuda())
            unstylized_audio_recon2 = self.model2(encoder_outputs_unstylized.permute(0, 2, 1).clone().cuda())
            
            # Compute loss
            loss = loss_fn(encoder_outputs_stylized, unstylized_audio_recon)
            loss2 = loss_fn(encoder_outputs_stylized, unstylized_audio_recon2.permute(0, 2, 1))
            
            # Backward pass
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            loss2.backward()
            optimizer.step()
            optimizer2.step()
            
            print(f"Epoch: {epoch} | Loss: {loss.item()} | Loss2: {loss2.item()}")
            
            
        output = self.encodec_model.decoder(unstylized_audio_recon.transpose(1, 2))[0]
        torchaudio.save("test.wav", output.cpu(), 24000)
        print()
