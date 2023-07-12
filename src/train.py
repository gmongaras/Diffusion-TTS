from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import numpy as np
import torch
import torchaudio
from TTS.api import TTS
from src.Model import Model





def train():
    # Get the TTS model
    model_name = 'tts_models/en/ljspeech/speedy-speech'
    tts = TTS(model_name, gpu=False)
    # Sampling rate
    sampling_rate = 22050
    
    
    # load the model + processor (for pre-processing the audio)
    encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz").eval().cuda()
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    
    # Get the quantizer from the model
    quantizer = encodec_model.quantizer
    
    
    
    
    
    
    #### Load in the dataset
    # First, load in the sample data
    stylized_audio, stylized_sr = torchaudio.load("notebooks/sample.wav")

    # Text for this audio
    text = "Mr. Quilter is the Apostle of the Middle Class and we are glad to welcome his gospel."
    
    # Generate an unstylized audio clip based off the same text
    unstylized_audio = torch.tensor(tts.tts(text)).float()
    sr = 22050
    
    # Batch the audio
    stylized_audio = stylized_audio.repeat(2, 1, 1)
    unstylized_audio = unstylized_audio.repeat(2, 1, 1)
    
    # Resample both audio clips to 24khz
    stylized_audio = torchaudio.transforms.Resample(stylized_sr, 24000)(stylized_audio).squeeze()
    unstylized_audio = torchaudio.transforms.Resample(sr, 24000)(unstylized_audio).squeeze()
        
    
    
    
    
    
    
    # Preporcess the audio segments
    stylized_audio = processor(stylized_audio.tolist(), sampling_rate=24000, return_tensors="pt")
    unstylized_audio = processor(unstylized_audio.tolist(), sampling_rate=24000, return_tensors="pt")
    
    
    # Encode inputs
    encoder_outputs_stylized = encodec_model.encode(stylized_audio["input_values"].cuda(), stylized_audio["padding_mask"].cuda(), bandwidth=24.0)
    encoder_outputs_unstylized = encodec_model.encode(unstylized_audio["input_values"].cuda(), unstylized_audio["padding_mask"].cuda(), bandwidth=24.0)

    # Decode inputs
    stylized_audio_recon = encodec_model.decode(encoder_outputs_stylized.audio_codes, encoder_outputs_stylized.audio_scales, stylized_audio["padding_mask"].cuda())[0]
    unstylized_audio_recon = encodec_model.decode(encoder_outputs_unstylized.audio_codes, encoder_outputs_unstylized.audio_scales, unstylized_audio["padding_mask"].cuda())[0]
    encoder_outputs_stylized = encoder_outputs_stylized.audio_codes
    encoder_outputs_unstylized = encoder_outputs_unstylized.audio_codes
    
    
    
    # Dequantize the outputs.
    # Note the quantizer expects inputs to be of shape (CB, B, T)
    encoder_outputs_stylized = quantizer.decode(encoder_outputs_stylized.squeeze().transpose(0, 1))
    encoder_outputs_unstylized = quantizer.decode(encoder_outputs_unstylized.squeeze().transpose(0, 1))
    
    
    
    
    
    
    
    
    
    # # Random audio for time being
    # encoder_outputs_stylized = torch.tensor(np.random.randint(0, 1024, (2, 1, 32, 440)))
    # encoder_outputs_unstylized = torch.tensor(np.random.randint(0, 1024, (2, 1, 32, 423)))
    
    
    # Switching to regression task. Instead use the output of the quantizer
    # encoder_outputs_stylized = torch.randn(2, 1, 128, 440)
    # encoder_outputs_unstylized = torch.randn(2, 1, 128, 423)
    
    
    
    
    
    
    
    
    # Transpose the inputs to (B, T, E)
    encoder_outputs_stylized = encoder_outputs_stylized.squeeze().transpose(1, 2).float().cuda()
    encoder_outputs_unstylized = encoder_outputs_unstylized.squeeze().transpose(1, 2).float().cuda()
    
    # Normalize the embeddings between 0 and 1
    # encoder_outputs_stylized /= 1024
    # encoder_outputs_unstylized /= 1024
    
    # Input conditiong text
    # (B, T_S, 32)
    text = torch.randn(2, 256, 32)
    
    
    # Transformer (B, T, 32) -> (B, T, 32)
    # model = Model(32).cuda()
    
    
    # Create the model
    from src.models.Transformer import Transformer
    model = Transformer(128, 128, 4).cuda()
    
    # Optimzer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    # Loss function
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(0, 10000):
        # Pad the unstlyized audio to the same length as the stylized audio
        padding_pre = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1])//2
        padding_post = (encoder_outputs_stylized.shape[1] - encoder_outputs_unstylized.shape[1]) - padding_pre
        encoder_outputs_unstylized = torch.nn.functional.pad(encoder_outputs_unstylized, (0, 0, padding_pre, padding_post))
        
        # Forward pass
        unstylized_audio_recon = model(encoder_outputs_unstylized.cuda(), text.cuda())
        
        # Compute loss
        loss = loss_fn(encoder_outputs_stylized, unstylized_audio_recon)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch: {epoch} | Loss: {loss.item()}")
    
    # output = get_output(
    #     encoder_outputs_unstylized, encoder_outputs_unstylized.audio_scales, unstylized_audio["padding_mask"].cuda()
    # )
    
    output = encodec_model.decoder(unstylized_audio_recon.transpose(1, 2))[0]
    torchaudio.save("test.wav", output.cpu(), 24000)
    print()





if __name__ == "__main__":
    train()