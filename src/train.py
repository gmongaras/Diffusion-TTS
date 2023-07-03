import torch
import datasets

import sys
sys.path.append('../')
from .encodec.model import EncodecModel
from .encodec.modules.seanet import SEANetEncoder, SEANetDecoder
from .encodec.quantization.vq import ResidualVectorQuantizer
from .encodec.balancer import Balancer
from .losses import Losses

from audio_dataset.make_hugginface import load_dataset



# Params
data_path = "audio_dataset/data"
device = torch.device("cuda:0")




def main():
    # Load in the dataset
    dataset = datasets.load_dataset(data_path, split="train")
    
    # Batch the dataset
    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}
    train_ds = dataset.map(group_batch, batched=True, batch_size=16)
    
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
    ).to(device)
    
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
    for i, batch in enumerate(train_ds):
        # Get the batch data
        audio = [torch.tensor(i["array"]) for i in batch["audio"]]
        
        # Pad the batch data to the max length audio data in the batch
        max_len = max([len(i) for i in audio])
        audio = torch.stack([torch.cat([i, torch.zeros(max_len - len(i))]) for i in audio])
        audio = torch.nn.Parameter(audio.unsqueeze(1).to(device))
        
        # Send data through the model
        outputs = Model(audio)
        
        # Get the losses
        losses = losses_obj.get_losses(outputs, audio)
        
        # Step the loss balancer on the losses to do a backward pass
        loss_balancer.backward(
            losses,
            audio
        )
        
        # Step optimizer
        optim.step()
        optim.zero_grad()
        
        # Loss for debugging
        loss = torch.tensor([
            losses[k]*loss_weights[k] for k in losses.keys()
        ]).sum().cpu().item()
        
        print(f"Loss {loss:.4f}")



if __name__ == "__main__":
    main()