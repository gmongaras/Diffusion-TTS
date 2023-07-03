import torch
import torchaudio













class Losses():
    def __init__(self, sr=16000, num_bins=64, spectogram_scales=[5,6,7,8,9,10,11], device=torch.device("cpu")):
        self.scales = spectogram_scales
        
        # Create the spectrogram transform for each scale
        self.spectrogram_transforms = [
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=2**scale,
                hop_length=2**scale // 4,
                n_mels=num_bins,
            ).to(device) for scale in self.scales
        ]

    # Basic L1 and L2 loss
    def L1_Loss(self, x, y):
        return torch.abs(x - y).mean()

    def L2_Loss(self, x, y):
        return ((x - y)**2).mean()




    # Reconstruction loss for the waveform
    def waveform_reconstruction_loss(self, outputs, inputs):
        return self.L1_Loss(outputs, inputs)

    # Reconstruction loss for the spectrogram
    def spectrogram_reconstruction_loss(self, outputs, inputs):
        total_loss = 0
        
        # Iterate over the spectogram transforms
        for transforms in self.spectrogram_transforms:
            # Transform the input and outptus
            input_spectrogram = transforms(inputs)
            output_spectrogram = transforms(outputs)
            
            # Compute the L1 and L2 losses
            l1_loss = self.L1_Loss(input_spectrogram, output_spectrogram)
            l2_loss = self.L2_Loss(input_spectrogram, output_spectrogram)
            
            # Compute the loos combination where alpha=1
            total_loss += l1_loss + l2_loss

        return total_loss / len(self.scales)


    # Get dictionary of losses given inputs and outputs of the AE
    def get_losses(self, outputs, inputs):
        return {
            "wav_recon": self.waveform_reconstruction_loss(outputs, inputs),
            "spec_recon": self.spectrogram_reconstruction_loss(outputs, inputs)
        }