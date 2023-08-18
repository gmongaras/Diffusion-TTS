import torch
import torchaudio
from src.Model import Model




def infer():
    # checkpoint_path = "checkpoints/step_132000/"
    checkpoint_path = "checkpoints_audio/step_40000"
    # text = 'I wanted to know more about the subject, so I asked about it.'
    text = 'Her mother played on the piano and the young lady on the violin.'
    condition_paths = ["audio_stylized_speaker/6746/1.wav", "audio_stylized_speaker/6746/2.wav"]
    # condition_paths = ["my_voice/my_voice_1.wav", "my_voice/my_voice_2.wav"]
    num_steps = 25
    outfile = "output.wav"
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    
    
    
    # Load in the model
    embed_dim = 256
    t_embed_dim = 256
    cond_embed_dim = 128
    num_blocks = 3
    blk_types = [
        ["res", "cond3", "ctx", "res"],
        ["res", "atn", "cond3", "ctx", "res"],
        ["res", "atn", "cond3", "ctx", "res"],
    ]
    use_noise = True
    noise_scheduler_type = "cosine"
    model = Model(embed_dim=embed_dim, 
                  t_embed_dim=t_embed_dim,
                  cond_embed_dim=cond_embed_dim, 
                  num_blocks=num_blocks,
                  blk_types=blk_types,
                  noise_scheduler_type=noise_scheduler_type,
                  device=device,
                  use_noise=use_noise, 
                  optim_8bit=False
            )
    
    # Load in the checkpoint
    optimizer_checkpoint = model.load_checkpoint(checkpoint_path)
    
    # Run model inference
    output = model.infer(text, condition_paths, num_steps=num_steps)
    
    # Save the output
    torchaudio.save(outfile, output, 24000)






if __name__ == "__main__":
    infer()