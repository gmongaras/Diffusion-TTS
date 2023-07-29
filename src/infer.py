import torch
import torchaudio
from src.Model import Model




def infer():
    checkpoint_path = "checkpoints_cond2_new/epoch_10/"
    text = "I agree with you, but I hate you!"
    condition_paths = ["audio_stylized_speaker/29/1.wav", "audio_stylized_speaker/29/2.wav"]
    num_steps = 100
    outfile = "output.wav"
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    
    
    
    # Load in the model
    model = Model(device)
    
    # Load in the checkpoint
    optimizer_checkpoint = model.load_checkpoint(checkpoint_path)
    
    # Run model inference
    output = model.infer(text, condition_paths, num_steps=num_steps)
    
    # Save the output
    torchaudio.save(outfile, output, 24000)






if __name__ == "__main__":
    infer()