import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

import sys
sys.path.append('../')
from .encodec.model import EncodecModel
from .encodec.modules.seanet import SEANetEncoder, SEANetDecoder
from .encodec.quantization.vq import ResidualVectorQuantizer
from .encodec.balancer import Balancer
from .losses import Losses



# Params
data_path = "audio_dataset/data"
device = torch.device("cuda:0")
epochs = 100
batch_size=8
num_workers=0




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
        
        # Breakup the audio into overlapping segments of 1 second
        # Overlap by 0.25 seconds
        if waveform.shape[1] > 16000:
            waveform = waveform.unfold(-1, 16000, 12000).transpose(0, 1)
        else:
            # Pad waveform to 16000
            waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1])).unsqueeze(0)
        
        return waveform




def main():
    # Create the WAVDataset
    dataset = WavDataset(data_path)
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers, 
                            collate_fn=lambda x: x
    )

    
    # Create the model
    dim = 128
    encoder, decoder, vq = SEANetEncoder(dimension=dim), SEANetDecoder(dimension=dim), ResidualVectorQuantizer(dim)
    Model = EncodecModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=vq,
        target_bandwidths=[6.0, 8.0, 10.0],
        sample_rate=16000,
        channels=1
    ).to(device).train()
    
    # Class to get all losses
    losses_obj = Losses(device=device)
    
    # Loss weights (4.4 in paper)
    loss_weights = {
        "wav_recon": 0.1,
        "spec_recon": 1.0,
    }
    
    # Used for loss balancing
    loss_balancer = Balancer(
        loss_weights
    )
    
    # Optimizer
    optim = torch.optim.AdamW(Model.parameters(), lr=3e-4, betas=(0.5, 0.9))
    
    # Iterate over the dataset
    for epoch in range(epochs):
        total = 0
        for i, batch in enumerate(dataloader):
            # Concatenate the batch along the time dimension
            # (N, T, 1, 16000) -> (N*T, 1, 16000)
            audio = torch.nn.Parameter(torch.cat(batch).detach().to(device))
            
            # Send data through the model
            outputs = Model(audio)
            
            # Get the losses
            losses = losses_obj.get_losses(outputs, audio)
            
            # Step the loss balancer on the losses to do a backward pass
            loss_balancer.backward(
                losses,
                outputs
            )
            
            # Step optimizer
            optim.step()
            optim.zero_grad()
            
            # Loss for debugging
            loss = sum([
                losses[k]*loss_weights[k] for k in losses.keys()
            ])
            
            # print(f"Loss {loss:.4f}")
            total += loss
            
            del audio, batch, outputs, losses, loss
        
        print(f"Epoch {epoch} Loss {total / len(dataloader)}")



if __name__ == "__main__":
    main()