"""
[WiP] Training script for LiquidAI/LFM2-Audio-1.5B
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm
import os

from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
from liquid_audio.utils import LFMModality, mel2emb_len

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def audio_to_mel(audio):
    """Convert audio to mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128
    )
    audio = audio.float()  # Ensure float32
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    mel = mel_transform(audio).squeeze(0)
    return torch.log(mel + 1e-5)


def prepare_batch(audio, text, processor, device, dtype):
    """Prepare one training example."""
    
    # Convert audio to mel
    mel = audio_to_mel(audio)
    mel_len = mel.shape[1]
    
    # Tokenize
    sys_tokens = processor.text.encode("Perform ASR.", add_special_tokens=True)
    tgt_tokens = processor.text.encode(text, add_special_tokens=True)
    
    # Modality flag
    mel_emb_len = mel2emb_len(torch.tensor([mel_len])).item()
    modality_flag = torch.cat([
        torch.full((len(sys_tokens),), LFMModality.TEXT),
        torch.full((mel_emb_len,), LFMModality.AUDIO_IN),
        torch.full((len(tgt_tokens),), LFMModality.TEXT),
    ]).unsqueeze(0)
    
    # Text and labels
    text_tensor = torch.tensor(sys_tokens + tgt_tokens).unsqueeze(0)
    labels = torch.full((1, modality_flag.shape[1]), -100)
    tgt_start = len(sys_tokens) + mel_emb_len
    for i, tok in enumerate(tgt_tokens):
        labels[0, tgt_start + i] = tok
    
    return {
        'text': text_tensor.to(device),
        'audio_in': mel.to(device, dtype=dtype),
        'audio_in_lens': torch.tensor([mel_len]).to(device),
        'audio_out': torch.empty((8, 0), dtype=torch.long).to(device),
        'modality_flag': modality_flag.to(device),
        'labels': labels.to(device),
    }


def forward_pass(model, batch):
    """Forward pass and loss computation."""
    
    # Get embeddings
    in_emb = model._prefill(
        text=batch['text'],
        audio_in=batch['audio_in'],
        audio_in_lens=batch['audio_in_lens'],
        audio_out=batch['audio_out'],
        modality_flag=batch['modality_flag'],
    )
    
    # Forward
    out = model.lfm(inputs_embeds=in_emb)
    logits = F.linear(out.last_hidden_state[0], model.lfm.embed_tokens.weight)
    
    # Loss
    loss = F.cross_entropy(
        logits[:-1].reshape(-1, logits.size(-1)),
        batch['labels'][0, 1:].reshape(-1),
        ignore_index=-100,
    )
    
    return loss


def train(
    num_epochs=1,
    lr=5e-5,
    output_dir="./output",
    use_wandb=True,
    wandb_project="lfm2-audio-asr",
    wandb_run_name=None,
):
    """Main training function with W&B tracking."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "model": "LFM2-Audio-1.5B",
                "dtype": str(dtype),
                "device": str(device),
            }
        )
        print("✓ W&B initialized")
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠ W&B requested but not available. Install with: pip install wandb")
        use_wandb = False
    
    # Load model
    print("Loading model...")
    processor = LFM2AudioProcessor.from_pretrained("LiquidAI/LFM2-Audio-1.5B")
    model = LFM2AudioModel.from_pretrained(
        "LiquidAI/LFM2-Audio-1.5B",
        dtype=dtype,
        device=device,
    )
    
    # Fix component dtypes
    model.audio_embedding = model.audio_embedding.to(dtype=dtype)
    model.conformer = model.conformer.to(dtype=dtype)
    model.audio_adapter = model.audio_adapter.to(dtype=dtype)
    
    print("✓ Model loaded")
    
    # Log model info to W&B
    if use_wandb:
        num_params = sum(p.numel() for p in model.parameters())
        wandb.config.update({
            "num_parameters": num_params,
            "num_parameters_B": num_params / 1e9,
        })
    
    # Load data
    print("Loading data...")
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    print(f"✓ {len(dataset)} samples")
    
    if use_wandb:
        wandb.config.update({"num_samples": len(dataset)})
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Training loop
    print("\nTraining...")
    model.train()
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        progress = tqdm(dataset, desc="Training")
        
        for step, item in enumerate(progress):
            # Prepare batch
            audio = torch.tensor(item['audio']['array'], dtype=torch.float32)
            text = item.get('sentence') or item.get('text', '')
            batch = prepare_batch(audio, text, processor, device, dtype)
            
            # Forward + backward
            loss = forward_pass(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Track loss
            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1
            
            # Update progress bar
            progress.set_postfix({'loss': f'{loss_val:.4f}'})
            
            # Log to W&B
            if use_wandb:
                wandb.log({
                    "train/loss": loss_val,
                    "train/epoch": epoch + 1,
                    "train/step": global_step,
                })
            
            # Console logging
            if (step + 1) % 10 == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"  Step {step + 1}: loss={loss_val:.4f}, avg={avg_loss:.4f}")
                
                if use_wandb:
                    wandb.log({
                        "train/avg_loss": avg_loss,
                        "train/step": global_step,
                    })
        
        # End of epoch
        avg_epoch_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_epoch_loss,
                "train/epoch": epoch + 1,
            })
    
    # Save
    print(f"\nSaving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), f"{output_dir}/model.pt")
    processor.save_pretrained(output_dir)
    
    print(f"✓ Model saved to {output_dir}/model.pt")
    print(f"✓ Processor saved to {output_dir}")
    
    # Save model as W&B artifact
    if use_wandb:
        artifact = wandb.Artifact(
            name=f"lfm2-audio-asr-{wandb.run.id}",
            type="model",
            description="Fine-tuned LFM2-Audio model for ASR"
        )
        artifact.add_file(f"{output_dir}/model.pt")
        artifact.add_dir(output_dir)
        wandb.log_artifact(artifact)
        print("✓ Model saved to W&B")
        
        # Finish W&B run
        wandb.finish()
    
    print("✓ Done!")
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./output")
    
    # W&B arguments
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="lfm2-audio-asr", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    
    args = parser.parse_args()
    
    train(
        num_epochs=args.num_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
