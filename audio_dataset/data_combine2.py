# This script works with GigaSpeech
from datasets import load_dataset
import torch
import torchaudio
import numpy as np
import os



### Goal: We want a directory of wav files in the same format




# Params
resample_rate = 16000   # Resample audio rate
max_time = 10           # Max time of a single audio clip in seconds


# Directory to save all wavs to
out_dir = "/scratch/users/gmongaras/audio_stylized2"




if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Load in the dataset
gs = load_dataset("speechcolab/gigaspeech", "l", use_auth_token=True, cache_dir="/scratch/users/gmongaras/gigaspeech", data_dir="/scratch/users/gmongaras/gigaspeech")

# Iterate over all audio samples in the data
for split in gs.keys():
    # We only care about the audio data
    # data = gs[split]["audio"]

    for i, row in enumerate(gs.get(split)):
        # Get the audio and sampling rate
        audio = torch.tensor(row["audio"]["array"].astype(np.float32))
        sr = row["audio"]["sampling_rate"]

        # Make audio mono
        if len(audio.shape) > 1:
            audio = audio.mean(0, keepdim=True).to(torch.float32)
        else:
            audio = audio.unsqueeze(0)

        # Resample audio
        audio = torchaudio.transforms.Resample(sr, resample_rate)(audio)

        # If the audio is longer than max_time, we split it into multiple files
        if audio.shape[1] > max_time * resample_rate:
            num_splits = int(audio.shape[1] / (max_time * resample_rate))
            for j in range(num_splits):
                # Save file if it's longer than a second
                if audio[:, j * (max_time * resample_rate) : (j + 1) * (max_time * resample_rate)].shape[1] < resample_rate:
                    continue

                torchaudio.save(f"{out_dir}/{i}.wav", audio[:, j * (max_time * resample_rate) : (j + 1) * (max_time * resample_rate)], resample_rate)

            continue

        # Save the file to the output directory
        torchaudio.save(f"{out_dir}/{i}.wav", audio, resample_rate)


        if i % 1000 == 0:
            print(f"{i} files processed")