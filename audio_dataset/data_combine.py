import os
import torch
import torchaudio



### Goal: We want a directory of wav files in the same format




# Params
resample_rate = 16000   # Resample audio rate
max_time = 10           # Max time of a single audio clip in seconds
min_time = 1            # Min time of a single audio clip in seconds


# Directory to save all wavs to
out_dir = "audio_stylized_speaker"

# Two directories. One for VCTK and the other for LibriTTS
VCTK_dir = "VCTK-Corpus"
LibriTTS_dir = "LibriTTS_R"


# Create the output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Count of the number of files in the output directory
num_files = 0

# # First, we parse through the VCTK directory.
# for root, dirs, files in os.walk(VCTK_dir + "/wav48"):
#     for file in files:
#         if file.endswith(".wav"):
#             # Load in the file
#             audio, sr = torchaudio.load(os.path.join(root, file))

#             # Make audio mono
#             audio = audio.mean(0, keepdim=True).to(torch.float32)

#             # Resample audio
#             audio = torchaudio.transforms.Resample(sr, resample_rate)(audio)

#             # If the audio is longer than max_time, we split it into multiple files
#             if audio.shape[1] > max_time * resample_rate:
#                 num_splits = int(audio.shape[1] / (max_time * resample_rate))
#                 for i in range(num_splits):
#                     # Save file if it's longer than a second
#                     if audio[:, i * (max_time * resample_rate) : (i + 1) * (max_time * resample_rate)].shape[1] < resample_rate:
#                         continue

#                     torchaudio.save(f"{out_dir}/{num_files}.wav", audio[:, i * (max_time * resample_rate) : (i + 1) * (max_time * resample_rate)], resample_rate)
#                     num_files += 1

#                 continue

#             # Save the file to the output directory
#             torchaudio.save(f"{out_dir}/{num_files}.wav", audio, resample_rate)
#             num_files += 1
            

#             if num_files % 1000 == 0:
#                 print(f"{num_files} files processed")



# # Now, we parse through the LibriTTS directory.
# for root, dirs, files in os.walk(LibriTTS_dir):
#     for file in files:
#         if file.endswith(".wav"):
#             # Load in the file
#             audio, sr = torchaudio.load(os.path.join(root, file))

#             # Make audio mono
#             audio = audio.mean(0, keepdim=True).to(torch.float32)

#             # Resample audio
#             audio = torchaudio.transforms.Resample(sr, resample_rate)(audio)

#             # If the audio is longer than max_time, we split it into multiple files
#             if audio.shape[1] > max_time * resample_rate:
#                 num_splits = int(audio.shape[1] / (max_time * resample_rate))
#                 for i in range(num_splits):
#                     # Save file if it's longer than a second
#                     if audio[:, i * (max_time * resample_rate) : (i + 1) * (max_time * resample_rate)].shape[1] < resample_rate:
#                         continue

#                     torchaudio.save(f"{out_dir}/{num_files}.wav", audio[:, i * (max_time * resample_rate) : (i + 1) * (max_time * resample_rate)], resample_rate)
#                     num_files += 1

#                 continue

#             # Save the file to the output directory
#             torchaudio.save(f"{out_dir}/{num_files}.wav", audio, resample_rate)
#             num_files += 1
            

#             if num_files % 1000 == 0:
#                 print(f"{num_files} files processed")




# Now, we parse through the LibriTTS directory.
for root, dirs, files in os.walk(LibriTTS_dir):
    for file in files:
        if file.endswith(".wav"):
            # Get the speaker ID from the file name
            speaker_id = file.split("_")[0]

            # Make sure the speaker directory exists
            if not os.path.exists(f"{out_dir}/{speaker_id}"):
                os.makedirs(f"{out_dir}/{speaker_id}")

            # Get the number of files in the speaker directory
            file_num = len(os.listdir(f"{out_dir}/{speaker_id}"))//2
            #if file_num > 10:
            #    continue

            # Load in the text file and save it
            with open(os.path.join(root, file[:-4] + ".normalized.txt"), "r") as f:
                text = f.read()
                with open(f"{out_dir}/{speaker_id}/{file_num}.txt", "w", encoding="utf-8") as f_out:
                    f_out.write(text.strip())

            # Load in the file
            audio, sr = torchaudio.load(os.path.join(root, file))

            # Make audio mono
            audio = audio.mean(0, keepdim=True).to(torch.float32)

            # Resample audio
            audio = torchaudio.transforms.Resample(sr, resample_rate)(audio)
            
            # Skip audio less than the min time
            if audio.shape[1] < min_time*resample_rate:
                continue

            # If the audio is longer than max_time, we split it into multiple files
            if audio.shape[1] > max_time * resample_rate:
                num_splits = int(audio.shape[1] / (max_time * resample_rate))
                for i in range(num_splits):
                    # Save file if it's longer than a second
                    if audio[:, i * (max_time * resample_rate) : (i + 1) * (max_time * resample_rate)].shape[1] < resample_rate:
                        continue

                    torchaudio.save(f"{out_dir}/{speaker_id}/{file_num}.wav", audio[:, i * (max_time * resample_rate) : (i + 1)])
                    num_files += 1

                continue

            # Save the file to the output directory
            torchaudio.save(f"{out_dir}/{speaker_id}/{file_num}.wav", audio, resample_rate)
            num_files += 1


            if num_files % 1000 == 0:
                print(f"{num_files} files processed")