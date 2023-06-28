# RoVoLaD-name-subject-to-change-
RoVoLaD - Robot voice to stylized voice latent diffusion model


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
- Multilingual LibriSpeech (MLS) - https://www.openslr.org/94/
- Gigaspeech: https://github.com/SpeechColab/GigaSpeech

# Papers
- Stable Diffusion - https://arxiv.org/abs/2112.10752
- Meta Voicebox - https://ai.facebook.com/blog/voicebox-generative-ai-model-speech/
- Natural Speech 2 - https://speechresearch.github.io/naturalspeech2/

- Encodec - https://arxiv.org/abs/2210.13438
- MusicGen - https://arxiv.org/abs/2306.05284
