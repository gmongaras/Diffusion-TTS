try: # For distributed training
    from Model import Model
    from Trainer import Trainer
    from WavDataset import WavDataset
except ModuleNotFoundError:
    from src.Model import Model
    from src.Trainer import Trainer
    from src.WavDataset import WavDataset



def train():
    # Data params
    data_path = "../audio_datasets/audio_stylized_speaker"
    num_workers = 8
    prefetch_factor = 8
    limit = None
    augment_audio = True 
    
    # Model params
    device = "gpu" # "gpu" or "cpu"
    embed_dim = 304#256
    t_embed_dim = 512
    cond_embed_dim = 128
    num_blocks = 3
    # blk_types = [
    #     ["res", "res", "cond3", "ctx"],
    #     ["res", "atn", "res", "cond3", "ctx"],
    #     ["res", "atn", "res", "cond3", "ctx"],
    # ]
    blk_types = [
        ["res", "fullctx", "res", "fullctx"],
        ["res", "fullctx", "res", "fullctx"],
        ["res", "fullctx", "res", "fullctx"],
    ]
    use_noise = True
    noise_scheduler_type = "linear"
    prediction_strategy = "audio"               # "noise" to predict noise, "audio" to predict stylized audio
    text_encoder_type = "T5"                    # CLIP or T5
    optim_8bit = False
    
    # Training params
    batch_size = 32#64
    lr = 1e-4
    save_every_steps = 5000
    accumulation_steps = 1
    use_scheduler = True
    sample_dir = "audio_samples_audio"
    checkpoints_dir = "checkpoints_audio"
    use_amp = True
    
    # Loading params
    # pretrained_checkpoint_path = "checkpoints_noise_fullctx2/step_65000/"
    pretrained_checkpoint_path = None
    
    
    
    
    
    # Create the WAVDataset
    dataset = WavDataset(data_path, augment_audio=augment_audio, load_in_memory=False, use_noise=use_noise, limit=limit)
    
    
    
    
    # Create the main model
    model = Model(embed_dim=embed_dim, 
                  t_embed_dim=t_embed_dim,
                  cond_embed_dim=cond_embed_dim, 
                  num_blocks=num_blocks,
                  blk_types=blk_types,
                  noise_scheduler_type=noise_scheduler_type,
                  prediction_strategy=prediction_strategy,
                  text_encoder_type=text_encoder_type,
                  device=device,
                  use_noise=use_noise, 
                  optim_8bit=optim_8bit,
            )
    
    # Load in the checkpoint
    optimizer_checkpoint = None
    scheduler_checkpoint = None
    epoch_ckpt = None
    step_ckpt = None
    if pretrained_checkpoint_path:
        optimizer_checkpoint, scheduler_checkpoint, epoch_ckpt, step_ckpt = model.load_checkpoint(pretrained_checkpoint_path)
    
    
    
    
    # Train the model
    trainer = Trainer(
        model=model, 
        dataset=dataset, 
        dev=device,
        batch_size=batch_size, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor, 
        lr=lr, 
        save_every_steps=save_every_steps, 
        use_scheduler=use_scheduler, 
        accumulation_steps=accumulation_steps, 
        sample_dir=sample_dir, 
        checkpoints_dir=checkpoints_dir, 
        optimizer_checkpoint=optimizer_checkpoint, 
        scheduler_checkpoint=scheduler_checkpoint,
        use_amp=use_amp,
    )
    trainer.train(epoch_ckpt, step_ckpt)
    # model.train_model()
    
    
    
    
    




if __name__ == "__main__":
    train()
