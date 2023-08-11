import torch
import torchaudio
import torch.nn as nn
import os
import json

from transformers import EncodecModel, AutoProcessor
from TTS.api import TTS
import open_clip

try: # For distributed training
    from models.U_Net import U_Net
    from utils.Diffusion_Utils import Diffusion_Utils
except ModuleNotFoundError:
    from src.models.U_Net import U_Net
    from src.utils.Diffusion_Utils import Diffusion_Utils



### From StableDiffusion 2:
### https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/encoders/modules.py#L176
class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, 
                                pretrained=version, 
                                cache_dir="./CLIP_weights", 
                                device=torch.device("cpu"),
                                precision="fp32" if device != "cpu" else "fp32",
                )
        del model.visual
        self.model = model.to(device)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)




class Model(nn.Module):
    def __init__(self, embed_dim=128, t_embed_dim=128, cond_embed_dim=128, num_blocks=2, blk_types=["res", "cond2", "res"], noise_scheduler_type="linear", use_noise=False, device="cpu"):
        super(Model, self).__init__()
        
        self.device = device
        self.use_noise = use_noise
        self.sampling_rate = 24_000
        self.scale = 1
        
        self.embed_dim = embed_dim
        self.t_embed_dim = t_embed_dim
        
        
        
        
        # Important default parameters
        self.defaults = {
            "embed_dim": embed_dim,
            "t_embed_dim": t_embed_dim,
            "cond_embed_dim": cond_embed_dim,
            "num_blocks": num_blocks,
            "blk_types": blk_types,
            "noise_scheduler_type": noise_scheduler_type,
            "use_noise": use_noise,
        }
        
        # Convert the device to a torch device
        if type(device) is str:
            if device.lower() == "gpu":
                if torch.cuda.is_available():
                    dev = device.lower()
                    try:
                        local_rank = int(os.environ['LOCAL_RANK'])
                    except KeyError:
                        local_rank = 0
                    device = torch.device(f"cuda:{local_rank}")
                else:
                    dev = "cpu"
                    print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
                    device = torch.device('cpu')
            else:
                dev = "cpu"
                device = torch.device('cpu')
            self.device = device
            self.dev = dev
        else:
            self.device = device
            self.dev = "cpu" if device.type == "cpu" else "gpu"
        
        
        
        
        
        ### Encodec
        ### Note: Load as lsits to hide from parameters
        # load the model + processor (for pre-processing the audio)
        self.encodec_model = [EncodecModel.from_pretrained("facebook/encodec_24khz").eval().to(self.device)]
        for param in self.encodec_model[0].parameters():
            param.requires_grad = False
        self.processor = [AutoProcessor.from_pretrained("facebook/encodec_24khz")]
        # Get the quantizer from the model
        self.quantizer = [self.encodec_model[0].quantizer]
        
        
        
        ### CLIP model
        # self.CLIP, self.CLIP_tok = get_clip_model(self.device)
        self.CLIP = [FrozenOpenCLIPEmbedder(device=self.device, freeze=True, layer="last")]
        
        
        ### TTS model
        if not self.use_noise:
            model_name = 'tts_models/en/ljspeech/speedy-speech'
            self.tts = [TTS(model_name, gpu=False)]
        
        
        # Model to train
        # self.model = Transformer(128, 128, 512, 8).to(self.device)
        if type(blk_types[0]) == str:
            blk_types = [blk_types for _ in range(num_blocks)]
        assert len(blk_types) == num_blocks, "blk_types must be the same length as num_blocks"
        self.model = U_Net(128, 128, embed_dim, 1, num_blocks=num_blocks, blk_types=blk_types, cond_dim=cond_embed_dim, t_dim=t_embed_dim, c_dim=1024).to(self.device)
        
        # Diffusion model utility class
        self.diffusion_utils = Diffusion_Utils(t_embed_dim, scheduler_type=noise_scheduler_type)
        
        # Paramater counts
        print("U-Net model has {} parameters".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        
        
        
        
        
    def process_data(self, audio_segment, masks=None):
        # Preporcess the audio segments
        audio_segment = self.processor[0](audio_segment.squeeze(1).tolist(), sampling_rate=24_000, return_tensors="pt")
        
        # Change the masks
        if masks is not None:
            for i, m in enumerate(masks):
                audio_segment["padding_mask"][i][m:] = 0
        
        # Encode inputs
        # NOTE: Encodec is stupid and expects the masks to be 0 where we want to
        # keep the audio and 1 where we want to mask it out
        encoded_outputs = self.encodec_model[0].encode(audio_segment["input_values"].to(self.device), audio_segment["padding_mask"].to(self.device), bandwidth=24.0)
        encoded_outputs = encoded_outputs.audio_codes
        
        
        # Dequantize the outputs.
        # Note the quantizer expects inputs to be of shape (CB, B, T)
        encoded_outputs = self.quantizer[0].decode(encoded_outputs.squeeze(0).transpose(0, 1))
        
        # Transpose the inputs to (B, E, T)
        encoded_outputs = encoded_outputs.float().to(self.device)
        
        # Change the masks to be in the context of the
        # latent dimension and make it a full matrix.
        # This matrix will be zero where we mask
        # and 1 where we keep
        if masks is not None:
            masks_full = torch.ones(encoded_outputs.shape[0], 1, encoded_outputs.shape[2], dtype=torch.bool, device=self.device)
            for i, m in enumerate(masks):
                # New mask is basically quantized from 24_000 -> 75
                new_mask = m//(self.sampling_rate//self.quantizer[0].frame_rate)
                masks_full[i][:, new_mask:] = 0
                
        # Return masked outputs
        if masks is None:
            return encoded_outputs, None
        return encoded_outputs*masks_full, masks_full
    
    
    
    
    
    
    
    # Input:
    #   audio_super - Superposition of the audio of shape (N, E, T)
    #   conditional - (optional) Batch of encoded conditonal information
    #       of shape (N, E, T2)
    #   positional_embeddings - (optional) Batch of encoded t values for each 
    #       X value of shape (N, t_dim)
    #   text - (optional) Batch of encoded context information. This
    #             is the text conditioning information. Shape (N, c_dim)
    #   masks - (optional) Batch of masks for each X value
    #       of shape (N, 1, T)
    #   masks_cond - (optional) Batch of masks for each c value
    #       of shape (N, 1, T2)
    def forward(self, audio_super, conditional=None, positional_embeddings=None, text=None, masks=None, masks_cond=None):
        audio_super = audio_super.to(self.device)
        if type(conditional) == torch.Tensor:
            conditional = conditional.to(self.device)
        if type(positional_embeddings) == torch.Tensor:
            positional_embeddings = positional_embeddings.to(self.device)
        if type(text) == torch.Tensor:
            text = text.to(self.device)
        if type(masks) == torch.Tensor:
            masks = masks.to(self.device)
        if type(masks_cond) == torch.Tensor:
            masks_cond = masks_cond.to(self.device)
            
        return self.model(audio_super, conditional, positional_embeddings, text, masks, masks_cond)
            
            
            
            
    
    # Given text and a list of conditionals, generate the stylized audio
    @torch.no_grad()
    def infer(self, text, conditionals=[], num_steps=100):
        if not self.use_noise:
            # Create the unstylized audio
            try:
                unstylized = torch.tensor(self.tts[0].tts(text)).float()
            except RuntimeError:
                unstylized = torch.tensor(self.tts[0].tts(text + "...")).float()
                
            # Resample generated audio to 24000 Hz
            unstylized = torchaudio.transforms.Resample(22050, 24000)(unstylized)
            
            unstylized, _ = self.process_data(unstylized.unsqueeze(0))
        else:
            unstylized = torch.randn(1, 128, 300)
        
        # Load in the conditional audio
        conditionals = [torchaudio.load(path) for path in conditionals]
        conditionals = torch.cat([torchaudio.transforms.Resample(c[1], 24000)(c[0]) for c in conditionals], dim=1)
        
        # Encode the text data
        text = self.CLIP[0].encode(text)
        
        # Pass the data through the encodec
        # conditionals = [self.process_data(c)[0] for c in conditionals]
        conditionals, _ = self.process_data(conditionals)
        
        # Pad the unstylized audio to the nearest second
        # Note: 75 is a second in the latent dimension
        # unstylized = torch.nn.functional.pad(unstylized, (0, 75 - unstylized.shape[2]%75, 0, 0))
        
        # Concatenate the conditional audio along the time dimension
        # conditionals = torch.cat(conditionals, dim=2)
        
        # Permute audio to be of shape (N, E, T)
        unstylized = unstylized.to(self.device)
        conditionals = conditionals.to(self.device)
        
        # Get prediction
        pred = self.diffusion_utils.sample_data(self.model, unstylized, num_steps=num_steps, cond=conditionals, context=text)
        # pred *= self.scale
        
        # Decode the audio
        return self.encodec_model[0].decoder(pred.to(self.device))[0].cpu()
    
    
    
    
    
    # Used to load in checkpoints
    def load_checkpoint(self, path):
        # Load in paramaters
        with open(path + "/model_params.json", "r") as f:
            self.defaults = json.load(f)
        self.defaults["device"] = self.device
        
        step = self.defaults["step"]
        epoch = self.defaults["epoch"]
        del self.defaults["step"]
        del self.defaults["epoch"]
            
        # Reinit model with checkpoint
        self.__init__(**self.defaults)
        
        # Load in the model
        self.load_state_dict(torch.load(path + "/model.pth", map_location=self.device))
        self.eval()
        
        # Load in the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        try:
            optimizer.load_state_dict(torch.load(path + "/optimizer.pth", map_location=self.device))
        except:
            print("Optimizer checkpoint not found")
            optimizer = None
        
        # Load in the scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6)
        try:
            scheduler.load_state_dict(torch.load(path + "/scheduler.pth", map_location=self.device))
        except:
            print("Scheduler checkpoint not found")
            scheduler = None
        
        return optimizer, scheduler, epoch, step
