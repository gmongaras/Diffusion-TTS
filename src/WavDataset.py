import torch
import torchaudio
import os
from torch.utils.data import Dataset
from TTS.api import TTS
import random
import contextlib




class WavDataset(Dataset):
    @torch.no_grad()
    def __init__(self, root_dir, load_in_memory=False, use_noise=False, limit=-1):
        if limit == None:
            limit = -1
        self.load_in_memory = load_in_memory
        self.use_noise = use_noise
        
        # List of all data as a three part tuple:
        # (wav file, text file, tts output)
        self.data = []
        
        # Map speaker to indices of their data
        self.speaker_map = {}
        
        # Map data index to speaker ID and data index for that speaker
        self.reverse_idx = []
        
        # Iterate over all speakers in the directory
        self.total = 0
        for speaker in os.listdir(root_dir):
            # Iterate over all the audio files
            for f in os.listdir(os.path.join(root_dir, speaker)):
                # If the file is a wav file, add it to the dataset along with its corresponding text file
                if f.endswith('.wav'):
                    # Limit each speaker
                    if limit != -1 and len(self.speaker_map.get(speaker, [])) >= limit:
                        continue
                    
                    if not speaker in self.speaker_map:
                        self.speaker_map[speaker] = []
                    
                    # Data index to speaker and index in speaker
                    self.reverse_idx.append((speaker, len(self.speaker_map[speaker])))
                    
                    # Speaker to index in data
                    self.speaker_map[speaker].append(self.total)
                    
                    # Add data to list
                    self.data.append((
                        os.path.join(root_dir, speaker, f),
                        os.path.join(root_dir, speaker, f.replace('.wav', '.txt'))
                    ))
                    
                    self.total += 1
                    
        # Load data into memory
        if self.load_in_memory:
            self.data = [(torchaudio.load(f), open(txt, 'r').read().strip()) for f, txt in self.data]
            
        # Only load TTS if not using noise
        if not use_noise:
            # Load in the TTS model
            # model_name = 'tts_models/en/ljspeech/speedy-speech'
            # self.tts_sr = 22050
            model_name = 'tts_models/en/ljspeech/glow-tts'
            self.tts = TTS(model_name, gpu=False)
            self.tts_sr = self.tts.synthesizer.output_sample_rate
            
            
            # Transcribe the text using TTS
            if self.load_in_memory:
                self.data = [(waveform, text, self.tts.tts(text)) for waveform, text in self.data]
        
    @torch.no_grad()
    def __len__(self):
        return self.total
    
    @torch.no_grad()
    def load_item(self, idx, only_wav=False):
        # Get the data entry
        data = self.data[idx]
        
        # Get the waveform and sample rate from memory
        if self.load_in_memory:
            raise NotImplementedError
            
            waveform, sample_rate = data[0]
            
            if only_wav:
                return waveform
            
            text = data[1]
            waveform_unstylized, sample_rate_unstylized = (data[2], self.tts_sr)
        else:
            waveform, sample_rate = torchaudio.load(data[0])
            
            # Resample
            if sample_rate != 24000:
                waveform = torchaudio.transforms.Resample(sample_rate, 24000)(waveform)
            
            if only_wav:
                return waveform
            
            text = open(data[1], 'r', encoding="utf-8").read().strip()
            
            # Only get alpha numeric characters
            text = ''.join([c for c in text if c.isalnum() or c == ' ']) + '.'
            
            if self.use_noise:
                waveform_unstylized = None
            else:
                with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                    waveform_unstylized, sample_rate_unstylized = (torch.tensor(self.tts.tts(text)).float().unsqueeze(0), self.tts_sr)
        
        if not self.use_noise:
            # Resample audio to 24000 Hz
            waveform_unstylized = torchaudio.transforms.Resample(sample_rate_unstylized, 24000)(waveform_unstylized)
        
        return waveform, waveform_unstylized, text
    

    @torch.no_grad()
    def __getitem__(self, idx):
        # Load in the data
        waveform, waveform_unstylized, text = self.load_item(idx)
        
        # Get the speaker and index in speaker
        speaker, speaker_idx = self.reverse_idx[idx]
        
        # Get another random data entry from the same speaker, but
        # with a different index value.
        indices = self.speaker_map[speaker].copy()
        indices.remove(idx)
        conditional_waveform = None
        if len(indices) > 0:
            # Random number of indices between 1 and 3
            number_of_indices = random.randint(1, min(3, len(indices)))
            
            # Get the conditional audio
            for i in range(number_of_indices):
                # Get the new index and remove it so it can't be used again
                idx2 = random.choice(indices)
                indices.remove(idx2)
                
                # Load in the waveform
                waveform2 = self.load_item(idx2, only_wav=True)
                
                # Concatenate the new waveform to the current
                # conditional waveform
                if conditional_waveform is None:
                    conditional_waveform = waveform2
                else:
                    conditional_waveform = torch.cat((conditional_waveform, waveform2), dim=1)
        else:
            waveform2 = None
        
        
        # # Breakup the audio into overlapping segments of 1 second
        # # Overlap by 0.25 seconds
        # if waveform.shape[1] > 16000:
        #     waveform = waveform.unfold(-1, 16000, 12000).transpose(0, 1)
        # else:
        #     # Pad waveform to 16000
        #     waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1])).unsqueeze(0)
        
        return waveform, waveform_unstylized, text, conditional_waveform