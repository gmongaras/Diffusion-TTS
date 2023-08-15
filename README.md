# RoVoLaD-name-subject-to-change-
RoVoLaD - Robot voice to stylized voice latent diffusion model


# Tests and Ablations
1. Does the model work as claimed?
    - Train a model on a Gaussian prior and another on the proposed robot prior. Test the following:
        1. Accuracy/fidelity
        2. Training/convergence speed
        3. # of diffusion steps to generate data
    - Also train on both? Perhaps this could increase varity.
2. Is a CNN or transformer better?
    - Test both a transformer in the (T, E) domain and (E, T) domain.
3. Test padding:
    - Post padding after the sequence
    - Pre padding before the sequence
    - Padding before and after the sequence, evenly
4. Test conditioning:
    - Train on only conditioning (current method) so that the model has context of the desired style
    - Train on both conditioning and no conditioning (with like 10% probability).
        - This method allows for classifier free guidance.
        - Perhaps training with a set of data without conditioning (one off data from GigaSpeech) along
          with conditioning data results in better results and more variety?


# To-do
1. [ ] Collect dataset for AE training
    1. We need a stylized audio dataset.
    2. Robot audio dataset should be the stylized audio dataset translated to robot audio.
1. [ ] Train an AE using stylized audio data.
2. [ ] Train an AE using robot audio data.
    - Robot audio data has the same dimensionality as the stylized audio.
3. [ ] Create a latent diffusion model in the latent space of the AE.
    - Forward process transforms stylized audio latents into robot audio latents
    - Reverse process using the LDM transforms robot audio latents into stylized audio latents
4. [ ] Condition reverse LDM on audio samples
    - Can easily be done by using different samples from the same speaker


# Datasets
The main goal is to produce speech in anyone's voice. So, we need as diverse of a dataset as possible. Diversity is prioritized over quantity.
1. VCTK: [https://huggingface.co/datasets/vctk](https://datashare.ed.ac.uk/handle/10283/2651)
2. Multilingual LibriSpeech (MLS) - https://www.openslr.org/94/
3. Improved LibriTTS - https://openslr.org/141/
4. Gigaspeech: https://github.com/SpeechColab/GigaSpeech

# Downloading Data
1. Download LibriTTS data from here https://www.openslr.org/resources/141/train_other_500.tar.gz
   - Unzip into `LibriTTS_R/train-other-500/...`
2. Download VCTK data from https://datashare.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2
   - Unzip into `VCTK-corpus/...`
3. Run data_combine.py to create a new directory named `audio_stylized_speaker` with the data in it
4. Download all of GigaSpeech to `GigaSpeech_Data/`

# Datasets Sampling
Since we are going to be using multiple datasets, we need to sample in a way that doesn't bias toward certain speakers:

For each epoch:
1. Sample N random speakers from the dataset
2. Sample a few random utterances from each speaker
3. This is now a batch for a single training step.

# Papers
- Stable Diffusion - https://arxiv.org/abs/2112.10752
- Meta Voicebox - https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/
- Natural Speech 2 - https://speechresearch.github.io/naturalspeech2/
- YourTTS - https://arxiv.org/abs/2112.02418

- Encodec - https://arxiv.org/abs/2210.13438
- MusicGen - https://arxiv.org/abs/2306.05284

Papers for samplers:
- Main paper: https://arxiv.org/pdf/2206.00927.pdf
- Improvement: https://arxiv.org/pdf/2211.01095.pdf
- Reference (notation): https://arxiv.org/pdf/2006.11239.pdf

Schedulers:
- Main paper: https://arxiv.org/pdf/2301.10972.pdf
- Perhaps this can imporve the schedulers: https://arxiv.org/pdf/2305.08891.pdf






# Large Problems and Solutions
1. Issues with unstylized prior
    - The unstylized prior was my human intuition thinking it would be easier to go from a static prior with
      lots of information than a random prior.
    - This intuition was wrong and it turns out a Gaussian prior works way better
    - I'm guessing this is because a Gaussian prior has the lowest amount of information out of
      any prior while the unstylized has lots of information. The model probably didn't use
      any of it, leading to issues.
2. How should I condition the model?
    - There's three ways I see of conditioning the model when given an embedding of shape (N, E, T) and
      context of shape (N, E, T2) where T != T2 necessarily. All three use attention.
        1. The first is by having two attention mechanisms. The first transforms the embeddings to
           (N, E, T2) and the seocnd transforms it back to (N, E, T). This is very large, but
           also probably has the most modeling capacity
        2. The second is by having the query and k be the conditioning which find correlations
           along the embedding dimension rather than the time dimension. This results in an attention
           matrix of shape (N, E, E) which is then applied along the embedding dimension of the embeddings.
        3. The last is how stable diffusion does it by having the keys and values be the context
           while the queries are the embeddings. This feels unnatrual, but abviously works.
3. Should text be added to the model?
    - Using CLIP, we can use one of the methods from above to apply CLIP latents.







# Windows Problems and solutions:
1. TTS has the following error:
   `'charmap' codec can't encode character '\u026a' in position 2: character maps to <undefined>`
- Easy fix as this is just stupid print issues TTS has. In `TTS/tts/utils/text/tokenizer.py:TTSTokenizer:encode` comment out or delete the print lines:
  ```
  def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    # print(text)
                    # print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids
  ```
 
 Also there's a lot of annoying ou

2. If the Dataload is having pickling issues, it's probably due to the lambda function of local function being passed into the collate_function argument. To fix this, make a global function that's passed into the collate_function. Note that this cannot be a Lambda or a local function.