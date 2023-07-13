from datasets import load_dataset, Audio
import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader
from src.Model import Model




class WavDataset(Dataset):
    def __init__(self, root_dir, load_in_memory=False):
        self.load_in_memory = load_in_memory
        self.file_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.wav')]
        if self.load_in_memory:
            self.data = [torchaudio.load(f) for f in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the waveform and sample rate from memory
        if self.load_in_memory:
            waveform, sample_rate = self.data[idx]
        else:
            waveform, sample_rate = torchaudio.load(self.file_list[idx])
        
        # Resample audio to 16000 Hz
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)
        
        # # Breakup the audio into overlapping segments of 1 second
        # # Overlap by 0.25 seconds
        # if waveform.shape[1] > 16000:
        #     waveform = waveform.unfold(-1, 16000, 12000).transpose(0, 1)
        # else:
        #     # Pad waveform to 16000
        #     waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1])).unsqueeze(0)
        
        return waveform






def train():
    data_path = "audio_dataset/data"
    batch_size = 2
    num_workers = 0
    
    
    
    
    
     # Create the WAVDataset
    dataset = WavDataset(data_path)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            collate_fn=lambda x: x
    )
    
    
    
    
    # Create the main model
    model = Model()
    
    
    
    
    # Train the model
    model.train(dataloader)
    
    
    
    exit()
    
    
    
    
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
    
    
    
    
    
    # output = get_output(
    #     encoder_outputs_unstylized, encoder_outputs_unstylized.audio_scales, unstylized_audio["padding_mask"].cuda()
    # )
    
    




if __name__ == "__main__":
    train()