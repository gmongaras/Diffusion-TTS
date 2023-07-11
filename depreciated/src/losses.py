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
        return torch.abs(x - y)

    def L2_Loss(self, x, y):
        return ((x - y)**2)




    # Reconstruction loss for the waveform
    def waveform_reconstruction_loss(self, outputs, inputs):
        return self.L1_Loss(outputs, inputs).mean()

    # Reconstruction loss for the spectrogram
    def spectrogram_reconstruction_loss(self, outputs, inputs):
        total_loss = 0
        
        # Iterate over the spectogram transforms
        for transforms in self.spectrogram_transforms:
            # Transform the input and outptus
            input_spectrogram = transforms(inputs)
            output_spectrogram = transforms(outputs)
            
            # Compute the L1 and L2 losses
            l1_loss = self.L1_Loss(input_spectrogram, output_spectrogram).mean()
            l2_loss = self.L2_Loss(input_spectrogram, output_spectrogram).mean()
            
            # Compute the loos combination where alpha=1
            total_loss += l1_loss + l2_loss

        return total_loss / len(self.scales) / input_spectrogram.shape[0]
    
    
    
    # Generator loss
    # 1/K * sum[max(0, 1-D(x_hat))]
    def generator_loss(self, disc_outputs_fake):
        total_loss = 0
        
        # Compute the loss for each discriminator
        for disc_output_fake in disc_outputs_fake:
            loss = torch.nn.functional.relu(1-disc_output_fake[0]).sum()
            total_loss += loss
            
        return total_loss / len(disc_outputs_fake) / disc_output_fake[0][0].shape[0]
    
    # Discriminator loss
    # 1/K * sum[max(0, 1-D(x)) + max(0, 1+D(x_hat))]
    def discriminator_loss(self, disc_outputs_real, disc_outputs_fake_det):
        total_loss = 0
        
        # Compute the loss for each discriminator
        for disc_output_real, disc_output_fake_det in zip(disc_outputs_real, disc_outputs_fake_det):
            loss = torch.nn.functional.relu(1-disc_output_real[0]).sum() + torch.nn.functional.relu(1+disc_output_fake_det[0]).sum()
            
            # 33% chance of not updating the discriminator
            if torch.rand(1) > 0.66666666:
                loss = loss*0
            
            total_loss += loss
            
        return total_loss / len(disc_outputs_real) / disc_output_real[0].shape[0]
    
    
    # Feature matching loss for the generator
    # This loss is basically the L1 loss between the discriminator
    # hidden layer outputs for the real and fake spectrograms
    # so the the generator learns to generate spectrograms that
    # are similar to the real ones even in the feature space
    # (1/KL) * sum_K[sum_L[  L1( D_lk(x), D_lk(x_hat) ) / mean(L1(D_lk(x)))  ]]
    #     where l is the layer index
    #           k is the discriminator index
    def feature_matching_loss(self, disc_outputs_real, disc_outputs_fake):
        total_loss = 0
        
        # Compute the loss for each discriminator
        for disc_output_real, disc_output_fake in zip(disc_outputs_real, disc_outputs_fake):
            disc_loss = 0
            
            # Compute loss for each layer
            for disc_output_real_layer, disc_output_fake_layer in zip(disc_output_real[1], disc_output_fake[1]):
                loss = self.L1_Loss(disc_output_real_layer, disc_output_fake_layer)
                
                # Normalize the loss by the mean of the L1 loss of the real outputs
                loss = loss / torch.norm(disc_output_real_layer, p=1, dim=-1).mean(-1).mean(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                
                disc_loss += loss.sum()
                
            total_loss += disc_loss
            
        return total_loss / len(disc_output_real) / len(disc_output_real[1]) / disc_output_real[0].shape[0]


    # Get dictionary of losses given inputs and outputs of the AE
    # outputs - Output of AE of shape (B, 1, 16000)
    # inputs - Input to AE of shape (B, 1, 16000)
    # disc_outputs_real - List of outputs of the STFT discriminators for the real spectrograms
    # disc_outputs_fake - List of outputs of the STFT discriminators for the fake spectrograms
    def get_losses(self, outputs, inputs, disc_outputs_real, disc_outputs_fake, disc_outputs_fake_det):
        return {
            "wav_recon": self.waveform_reconstruction_loss(outputs, inputs), # l_t
            "spec_recon": self.spectrogram_reconstruction_loss(outputs, inputs)*0, # l_f
            # "gen_loss": self.generator_loss(disc_outputs_fake), # l_g
            # "disc_loss": self.discriminator_loss(disc_outputs_real, disc_outputs_fake_det), # l_d
            # "feature_matching_loss": self.feature_matching_loss(disc_outputs_fake, disc_outputs_real), # l_feat
        }