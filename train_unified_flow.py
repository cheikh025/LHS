"""
Unified Multi-Task Normalizing Flow Training

Trains a SINGLE normalizing flow on embeddings from ALL tasks.
1. Load augmented.json from all task directories
2. Encode on-the-fly using SentenceTransformer
3. Train flow on combined embeddings

The unified flow can then be used by:
- Gradient-based search (gradient_evolution.py)
- Any task-specific search that uses prior-space operations

Key Advantages:
- Single model for all tasks
- Learns shared manifold structure across combinatorial optimization problems
- Enables cross-task knowledge transfer

Usage:
    python train_unified_flow.py
    python train_unified_flow.py --resume Flow_Checkpoints/unified_flow_epoch100.pth
"""

import os
import json
import glob
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from normalizing_flow import NormalizingFlow, compute_flow_loss
from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder
from utils import is_valid_python


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_augmented_heuristics(task_dir: str = "task") -> List[Tuple[str, str]]:
    """
    Load all augmented heuristics from all tasks.

    Returns:
        List of (code, task_name) tuples
    """
    all_heuristics = []
    augmented_files = glob.glob(os.path.join(task_dir, "*", "augmented.json"))

    print(f"\n{'='*70}")
    print(f"Loading Augmented Heuristics from All Tasks")
    print(f"{'='*70}")
    print(f"Found {len(augmented_files)} task directories\n")

    task_counts = {}

    for aug_file in sorted(augmented_files):
        task_name = Path(aug_file).parent.name

        try:
            with open(aug_file, 'r', encoding='utf-8') as f:
                heuristics_dict = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to load {aug_file}: {e}")
            continue

        if not heuristics_dict:
            warnings.warn(f"Empty augmented.json for task '{task_name}', skipping...")
            continue

        valid_count = 0
        for heuristic_name, code in heuristics_dict.items():
            if not code or not isinstance(code, str):
                continue
            if not is_valid_python(code):
                warnings.warn(f"Invalid Python syntax in {task_name}/{heuristic_name}, skipping...")
                continue
            all_heuristics.append((code, task_name))
            valid_count += 1

        task_counts[task_name] = valid_count
        print(f"  {task_name:25s}: {valid_count:3d} heuristics")

    print(f"\n{'='*70}")
    print(f"Total: {len(all_heuristics)} valid heuristics across {len(task_counts)} tasks")
    print(f"{'='*70}\n")

    return all_heuristics, task_counts


def encode_all_heuristics(
    heuristics_list: List[Tuple[str, str]],
    encoder_model: SentenceTransformer,
    device: str = "cuda",
    batch_size: int = 64
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Encode all heuristics using the code encoder.

    Returns:
        z_combined: Combined embeddings [total_programs, embedding_dim]
        metadata_df: DataFrame with task names
    """
    print(f"\n{'='*70}")
    print(f"Encoding All Heuristics")
    print(f"{'='*70}\n")

    codes = [h[0] for h in heuristics_list]
    tasks = [h[1] for h in heuristics_list]

    print(f"Encoding {len(codes)} programs in batches of {batch_size}...")
    embeddings = []

    encoder_model.eval()
    encoder_model.to(device)

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding"):
            batch_codes = codes[i:i+batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings).astype(np.float32)

    print(f"\n✓ Encoded all programs")
    print(f"  Embedding shape: {embeddings.shape}")

    metadata_df = pd.DataFrame({
        'task': tasks,
        'code': codes
    })

    print(f"\nDataset Statistics:")
    print(f"{'='*70}")
    task_counts = metadata_df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        print(f"  {task:25s}: {count:3d} programs")
    print(f"{'='*70}\n")

    return embeddings, metadata_df


def verify_embedding_distribution(z_combined: np.ndarray, metadata_df: pd.DataFrame):
    """Print statistics about the embedding distribution."""
    print(f"\n{'='*70}")
    print(f"Embedding Distribution Statistics")
    print(f"{'='*70}\n")

    print(f"Shape: {z_combined.shape}")
    print(f"Mean: {z_combined.mean():.4f}")
    print(f"Std: {z_combined.std():.4f}")
    print(f"Min: {z_combined.min():.4f}")
    print(f"Max: {z_combined.max():.4f}")

    print(f"\nPer-Task Distribution:")
    print(f"{'='*70}")
    task_counts = metadata_df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        percentage = count / len(z_combined) * 100
        print(f"  {task:25s}: {count:4d} embeddings ({percentage:5.1f}%)")
    print(f"{'='*70}\n")


# ============================================================================
# Checkpoint Management
# ============================================================================

def load_checkpoint_for_resume(
    checkpoint_path: str,
    flow_model: NormalizingFlow,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, Dict]:
    """Load checkpoint for resuming training."""
    print(f"\n{'='*70}")
    print(f"Loading Checkpoint for Resume Training")
    print(f"{'='*70}\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model state from: {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✓ Loaded optimizer state")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Loaded scheduler state")

    start_epoch = checkpoint.get('epoch', 0)

    print(f"\nCheckpoint Metadata:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Total programs: {checkpoint.get('total_programs', 'N/A')}")
    print(f"  Tasks: {len(checkpoint.get('tasks_trained', []))}")
    print(f"{'='*70}\n")

    return start_epoch, checkpoint


def save_checkpoint(
    epoch: int,
    flow_model: NormalizingFlow,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss: float,
    checkpoint_path: str,
    metadata: Optional[Dict] = None
):
    """Save training checkpoint with full state and metadata."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': flow_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'dim': flow_model.dim,
        'num_layers': flow_model.num_layers,
    }

    if metadata:
        checkpoint_data.update(metadata)

    torch.save(checkpoint_data, checkpoint_path)


# ============================================================================
# Training Function
# ============================================================================

def train_unified_flow(
    flow_model: NormalizingFlow,
    z_combined: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    batch_size: int = 128,
    epochs: int = 200,
    holdout_ratio: float = 0.1,
    device: str = 'cuda',
    verbose: bool = True,
    checkpoint_dir: str = "Flow_Checkpoints",
    start_epoch: int = 0,
    save_every: int = 20,
    metadata: Optional[Dict] = None
) -> NormalizingFlow:
    """
    Train unified normalizing flow on combined embeddings from all tasks.
    """
    from torch.utils.data import DataLoader, TensorDataset

    print(f"\n{'='*70}")
    print(f"Training: Unified Multi-Task Normalizing Flow")
    print(f"{'='*70}\n")

    # Prepare data with train/val split
    z_tensor = torch.tensor(z_combined, dtype=torch.float32)

    # Split into train and validation sets
    n_total = len(z_tensor)
    n_val = int(n_total * holdout_ratio)
    n_train = n_total - n_val

    # Shuffle indices for random split
    indices = torch.randperm(n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    z_train = z_tensor[train_indices]
    z_val = z_tensor[val_indices]

    print(f"Data Split:")
    print(f"  Total samples: {n_total}")
    print(f"  Training: {n_train} ({(1-holdout_ratio)*100:.1f}%)")
    print(f"  Validation: {n_val} ({holdout_ratio*100:.1f}%)")
    print()

    # Training DataLoader
    train_dataset = TensorDataset(z_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    # Validation DataLoader
    val_dataset = TensorDataset(z_val)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    # Move model to device
    flow_model = flow_model.to(device)
    flow_model.train()

    print(f"Training Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {start_epoch} → {epochs}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Total programs: {n_total} (train: {n_train}, val: {n_val})")
    print(f"  Batches per epoch: {len(train_dataloader)} (train), {len(val_dataloader)} (val)")
    print(f"  Device: {device}")
    print(f"  DataLoader workers: 8 (train), 4 (val)")

    if start_epoch > 0:
        print(f"\n  Resuming from epoch {start_epoch}")

    print(f"\nFlow Model Architecture:")
    print(f"  Dimension: {flow_model.dim}")
    print(f"  Layers: {flow_model.num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in flow_model.parameters()):,}")
    print(f"\nStarting training...\n")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, epochs):
        # ===== Training Phase =====
        flow_model.train()
        train_loss = 0.0
        num_train_batches = 0

        # Accumulate loss on GPU to avoid CPU sync per batch
        running_train_loss = torch.tensor(0.0, device=device)

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for (z_batch,) in pbar:
            z_batch = z_batch.to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad(set_to_none=True)

            # Compute flow loss
            loss = compute_flow_loss(flow_model, z_batch)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=5.0)

            # Update parameters
            optimizer.step()

            running_train_loss += loss.detach()
            num_train_batches += 1

            # Update progress bar
            if num_train_batches % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = running_train_loss.item() / max(num_train_batches, 1)

        # ===== Validation Phase =====
        flow_model.eval()
        running_val_loss = torch.tensor(0.0, device=device)
        num_val_batches = 0

        with torch.no_grad():
            for (z_batch,) in val_dataloader:
                z_batch = z_batch.to(device, non_blocking=True)
                loss = compute_flow_loss(flow_model, z_batch)
                running_val_loss += loss.detach()
                num_val_batches += 1

        avg_val_loss = running_val_loss.item() / max(num_val_batches, 1)

        # Update scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Track best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Best Val: {best_val_loss:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unified_flow_epoch{epoch+1}.pth")
            checkpoint_metadata = metadata.copy() if metadata else {}
            checkpoint_metadata.update({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            })
            save_checkpoint(
                epoch=epoch + 1,
                flow_model=flow_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,  # Save validation loss
                checkpoint_path=checkpoint_path,
                metadata=checkpoint_metadata
            )
            if verbose:
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save best model separately
        if avg_val_loss == best_val_loss:
            best_checkpoint_path = os.path.join(checkpoint_dir, "unified_flow_best.pth")
            checkpoint_metadata = metadata.copy() if metadata else {}
            checkpoint_metadata.update({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss,
                'is_best': True
            })
            save_checkpoint(
                epoch=epoch + 1,
                flow_model=flow_model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss=avg_val_loss,
                checkpoint_path=best_checkpoint_path,
                metadata=checkpoint_metadata
            )
            if verbose:
                print(f"  ✓ Best model saved: {best_checkpoint_path}")

    # Clear cache at end of training
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")

    return flow_model


# ============================================================================
# Validation Functions
# ============================================================================

def validate_flow(flow_model: NormalizingFlow, z_combined: np.ndarray, device: str = 'cuda'):
    """
    Validate the trained flow model.

    Checks:
    1. Reconstruction error (z → u → z should be identity)
    2. Prior distribution statistics (u should be ~ N(0, I))
    3. Log determinant statistics
    """
    print(f"\n{'='*70}")
    print(f"Validating Trained Flow Model")
    print(f"{'='*70}\n")

    flow_model.eval()
    flow_model = flow_model.to(device)

    # Sample for validation (use all data or subsample if too large)
    sample_size = min(10000, len(z_combined))
    indices = np.random.choice(len(z_combined), sample_size, replace=False)
    z_sample = torch.tensor(z_combined[indices], dtype=torch.float32, device=device)

    with torch.no_grad():
        # Forward pass
        u, log_det = flow_model(z_sample)

        # Inverse pass
        z_reconstructed = flow_model.inverse(u)

        # Compute metrics
        reconstruction_error = torch.mean((z_sample - z_reconstructed) ** 2).item()

        # Prior distribution statistics
        u_mean = u.mean(dim=0).mean().item()
        u_std = u.std(dim=0).mean().item()

        # Log determinant statistics
        log_det_mean = log_det.mean().item()
        log_det_std = log_det.std().item()

    print(f"Validation Results (on {sample_size} samples):")
    print(f"{'='*70}")
    print(f"  Reconstruction Error: {reconstruction_error:.6f} {'✓' if reconstruction_error < 1e-4 else '✗'}")
    print(f"  Prior Mean: {u_mean:.4f} (target: 0.0) {'✓' if abs(u_mean) < 0.1 else '✗'}")
    print(f"  Prior Std: {u_std:.4f} (target: 1.0) {'✓' if abs(u_std - 1.0) < 0.2 else '✗'}")
    print(f"  Log Det Mean: {log_det_mean:.4f}")
    print(f"  Log Det Std: {log_det_std:.4f}")
    print(f"{'='*70}\n")

    if reconstruction_error < 1e-4 and abs(u_mean) < 0.1 and abs(u_std - 1.0) < 0.2:
        print("✓ Flow model validation PASSED!")
    else:
        print("⚠ Flow model validation shows potential issues. Consider training longer.")
    print()


# ============================================================================
# Main Training Script
# ============================================================================

def main(resume_checkpoint: Optional[str] = None, holdout_ratio: float = 0.1, dropout: float = 0.1,
         encoder_name: str = None, embedding_dim: int = None):
    """Main training pipeline for unified normalizing flow.

    Args:
        resume_checkpoint: Path to checkpoint file to resume training from.
        holdout_ratio: Fraction of data to hold out for validation.
        dropout: Dropout probability for regularization.
        encoder_name: Encoder model name. Defaults to DEFAULT_ENCODER.
        embedding_dim: Matryoshka embedding dimension. If None, uses DEFAULT_MATRYOSHKA_DIM.
    """
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        print("TF32 matmul precision: enabled\n")

    # ========================================================================
    # Step 1: Load All Augmented Heuristics (Same as Unified Mapper)
    # ========================================================================

    heuristics_list, task_counts = load_all_augmented_heuristics(task_dir="task")

    if len(heuristics_list) == 0:
        raise ValueError("No valid heuristics loaded! Check task directories.")

    # ========================================================================
    # Step 2: Load Encoder Model
    # ========================================================================

    print(f"{'='*70}")
    print("Loading Encoder Model")
    print(f"{'='*70}\n")

    encoder_model, actual_embedding_dim = load_encoder(
        model_name=encoder_name,
        device=device,
        truncate_dim=embedding_dim
    )
    print(f"Encoder loaded ({encoder_name})")
    print(f"Embedding dimension: {actual_embedding_dim}\n")

    # ========================================================================
    # Step 3: Encode All Heuristics
    # ========================================================================

    z_combined, metadata_df = encode_all_heuristics(
        heuristics_list=heuristics_list,
        encoder_model=encoder_model,
        device=device,
        batch_size=64
    )

    # Free encoder memory
    del encoder_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Verify data distribution
    verify_embedding_distribution(z_combined, metadata_df)

    # ========================================================================
    # Step 4: Initialize Normalizing Flow
    # ========================================================================

    print(f"{'='*70}")
    print("Initializing Normalizing Flow")
    print(f"{'='*70}\n")

    embedding_dim = z_combined.shape[1]
    num_layers = 4  # Reduced from 8 for small datasets
    hidden_dim = 128  # Reduced from 512 for small datasets

    flow_model = NormalizingFlow(
        dim=embedding_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    num_params = sum(p.numel() for p in flow_model.parameters())
    print(f"Flow Model Architecture:")
    print(f"  Dimension: {embedding_dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout}")
    print(f"  Parameters: {num_params:,}")
    print()

    # ========================================================================
    # Step 5: Setup Optimizer and Scheduler
    # ========================================================================

    learning_rate = 1e-3
    optimizer = torch.optim.AdamW(
        flow_model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    # ========================================================================
    # Step 6: Load Checkpoint if Resuming
    # ========================================================================

    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch, _ = load_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint,
            flow_model=flow_model,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # ========================================================================
    # Step 7: Create Checkpoint Directory
    # ========================================================================

    checkpoint_dir = "Flow_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # Step 8: Prepare Metadata for Checkpoints
    # ========================================================================

    training_metadata = {
        'tasks_trained': list(task_counts.keys()),
        'total_programs': len(z_combined),
        'programs_per_task': task_counts,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'encoder': 'BAAI/bge-code-v1',
        'optimizer': 'AdamW',
        'initial_lr': learning_rate,
        'scheduler': 'ReduceLROnPlateau',
    }

    # ========================================================================
    # Step 9: Train Unified Flow
    # ========================================================================

    trained_flow = train_unified_flow(
        flow_model=flow_model,
        z_combined=z_combined,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=128,           # Flow is lighter, can use larger batches
        epochs=200,
        holdout_ratio=holdout_ratio,
        device=device,
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        save_every=20,            # Save every 20 epochs
        metadata=training_metadata
    )

    # ========================================================================
    # Step 10: Validate Trained Flow
    # ========================================================================

    validate_flow(trained_flow, z_combined, device=device)

    # ========================================================================
    # Step 11: Save Final Checkpoint
    # ========================================================================

    print(f"{'='*70}")
    print("Saving Final Checkpoint")
    print(f"{'='*70}\n")

    final_checkpoint = {
        'model_state_dict': trained_flow.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': 200,
        'dim': embedding_dim,
        'num_layers': num_layers,
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'tasks_trained': list(task_counts.keys()),
        'total_programs': len(z_combined),
        'programs_per_task': task_counts,
        'embedding_dim': embedding_dim,
        'encoder': 'BAAI/bge-code-v1',
        'architecture': 'RealNVP with ActNorm + Dropout',
        'training': {
            'optimizer': 'AdamW',
            'initial_lr': learning_rate,
            'weight_decay': 1e-5,
            'scheduler': 'ReduceLROnPlateau',
            'batch_size': 128,
            'epochs': 200,
            'holdout_ratio': holdout_ratio,
            'dropout': dropout,
            'gradient_clipping': 5.0,
        }
    }

    final_path = os.path.join(checkpoint_dir, "unified_flow_final.pth")
    torch.save(final_checkpoint, final_path)

    print(f"✓ Final model saved to: {final_path}")
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    print(f"  Total programs: {len(z_combined):,}")
    print(f"  Training split: {int((1-holdout_ratio)*len(z_combined))} train, {int(holdout_ratio*len(z_combined))} val")
    print(f"  Tasks trained: {len(task_counts)}")
    print(f"  Model parameters: {num_params:,}")
    print(f"  Architecture: RealNVP with ActNorm + Dropout")
    print(f"  Layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Holdout ratio: {holdout_ratio}")
    print(f"  Final checkpoint: {final_path}")
    print(f"  Best checkpoint: {os.path.join(checkpoint_dir, 'unified_flow_best.pth')}")
    print(f"{'='*70}\n")

    print("✓ Unified normalizing flow training complete!")
    print("\nYou can now use this flow for:")
    print("  1. Crossover evolution (crossover_evolution.py)")
    print("  2. Gradient-based search (gradient_evolution.py)")
    print("  3. Any prior-space operations across all tasks")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified multi-task normalizing flow training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--holdout_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to hold out for validation (default: 0.1)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for regularization (default: 0.1)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        help=f"Encoder model name (default: {DEFAULT_ENCODER})"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help=f"Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or 'model native'})"
    )
    args = parser.parse_args()

    main(
        resume_checkpoint=args.resume,
        holdout_ratio=args.holdout_ratio,
        dropout=args.dropout,
        encoder_name=args.encoder,
        embedding_dim=getattr(args, 'embedding_dim', None)
    )
