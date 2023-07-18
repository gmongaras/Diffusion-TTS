import torch
import torchaudio
from src.Model import Model




def infer():
    checkpoint_path = "checkpoints/epoch_11/"
    text = "I wonder if I should go to the store."
    condition_paths = ["audio_stylized_speaker/31/0.wav", "audio_stylized_speaker/31/3.wav"]
    num_steps = 100
    outfile = "output2.wav"
    
    
    
    # Load in the model
    model = Model()
    
    # Load in the checkpoint
    model.load_checkpoint(checkpoint_path)
    
    # Run model inference
    output = model.infer(text, condition_paths, num_steps=num_steps)
    
    # Save the output
    torchaudio.save(outfile, output, 24000)






if __name__ == "__main__":
    infer()