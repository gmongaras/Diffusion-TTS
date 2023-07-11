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
from .encodec.msstftd import DiscriminatorSTFT

from .utils import plot_waveform, plot_specgram



# Params
data_path = "audio_dataset/data"
device = torch.device("cuda:0")
epochs = 1000
batch_size=8
num_workers=0
accumulation_steps = 4




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
    
    # STFT Discriminators at different STFT window sizes
    # window_sizes = [2048, 1024, 512, 256, 128]
    windows_sizes = [256, 512]
    STFT_Discriminators = [DiscriminatorSTFT(
            filters=32,
            win_length=window_size,
            n_fft=window_size,
        ).to(device).train()
        for window_size in windows_sizes
    ]
    
    # Class to get all losses
    losses_obj = Losses(device=device)
    
    # Loss weights (4.4 in paper)
    loss_weights = {
        "wav_recon": 0.1,
        "spec_recon": 1.0,
        # "gen_loss": 3.0,
        # "feature_matching_loss": 3.0
    }
    
    # Used for loss balancing
    loss_balancer = Balancer(
        loss_weights
    )
    
    # Optimizers
    optim_gen = torch.optim.AdamW(
        Model.parameters(), 
        lr=3e-4, 
        betas=(0.5, 0.9)
    )
    disc_params = []
    for disc in STFT_Discriminators:
        disc_params += list(disc.parameters())
    optim_disc = torch.optim.AdamW(
        disc_params
    )
    
    # Iterate over the dataset
    num_steps = 0
    for epoch in range(epochs):
        total_loss_gen = 0
        total_loss_disc = 0
        total_loss_recon = 0
        for i, batch in enumerate(dataloader):
            # Concatenate the batch along the time dimension
            # (N, T, 1, 16000) -> (N*T, 1, 16000)
            # audio = torch.nn.Parameter(torch.cat(batch).detach().to(device))
            
            # Pad waveform to max length in batch
            max_size = max([waveform.shape[1] for waveform in batch])
            batch = torch.cat([torch.nn.functional.pad(waveform, (0, max_size - waveform.shape[1])).unsqueeze(0) for waveform in batch])
            audio = batch.detach().to(device)
            
            # Send data through the model
            outputs = Model(audio)
            
            # Send the generated and real data through the discriminators
            disc_outputs_real = None #[disc(audio) for disc in STFT_Discriminators]
            disc_outputs_fake = None #[disc(outputs) for disc in STFT_Discriminators]
            disc_outputs_fake_det = None #[disc(outputs.detach()) for disc in STFT_Discriminators]
            
            # Get the losses
            losses = losses_obj.get_losses(outputs, audio, disc_outputs_real, disc_outputs_fake, disc_outputs_fake_det)
            
            # The discriminator loss is independent of the other losses
            # disc_loss = losses.pop("disc_loss")
            
            # Step the loss balancer on the losses to do a backward pass
            loss_balancer.backward(
                losses,
                outputs
            )
            # disc_loss.backward()
            
            # Step optimizer
            if num_steps % accumulation_steps == 0:
                optim_gen.step()
                optim_disc.step()
                optim_gen.zero_grad()
                optim_disc.zero_grad()
            
            # Loss for debugging
            loss = sum([
                losses[k]*loss_weights[k] for k in losses.keys()
            ])
            
            # print(f"Gen Loss {loss:.4f}      Disc Loss {disc_loss:.4f}")
            total_loss_gen += loss
            # total_loss_disc += disc_loss
            total_loss_recon += losses["wav_recon"]**0.5
            num_steps += 1
            
            # del audio, batch, outputs, losses, loss
        
        # print(f"Epoch {epoch} Loss {(total_loss_gen / len(dataloader)):.4f}      Disc Loss {(total_loss_disc / len(dataloader)):.4f}")
        print(f"Epoch {epoch} Loss {(total_loss_gen / len(dataloader)):.4f}\nWav Recon: {total_loss_recon / len(dataloader)}")
        
        # Plot waveform and spectograms
        if epoch % 10 == 0:
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            plot_waveform(outputs[0][:32000], 16000, f"outputs/output_wav_{epoch}.png")
            plot_specgram(outputs[0][:32000], 16000, f"outputs/output_spec_{epoch}.png")
            plot_waveform(audio[0][:32000], 16000, f"outputs/orig_wav_{epoch}.png")
            plot_specgram(audio[0][:32000], 16000, f"outputs/orig_spec_{epoch}.png")



if __name__ == "__main__":
    main()