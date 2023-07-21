import torch
import torchaudio
from src.Model import Model




def infer():
    checkpoint_path = "checkpoints2/epoch_27/"
    text = "Curtain Lectures delivered in the course of Thirty Years by mrs Margaret Caudle, and suffered by Job, her Husband."
    condition_paths = ["audio_stylized_speaker/31/0.wav", "audio_stylized_speaker/31/3.wav"]
    num_steps = 10000
    outfile = "output2.wav"
    # device = torch.device("cuda:0")
    device = torch.device("cuda:0")
    
    
    
    # Load in the model
    model = Model(device)
    
    # Load in the checkpoint
    model.load_checkpoint(checkpoint_path)
    
    # Run model inference
    output = model.infer(text, condition_paths, num_steps=num_steps)
    
    # Save the output
    torchaudio.save(outfile, output, 24000)






if __name__ == "__main__":
    infer()