"""
Optimized Unified Multi-Task Mapper Training

Optimizations for RTX 6000 Pro (96GB VRAM):
- Batch size 8 with gradient accumulation (effective batch 16)
- Flash Attention 2 for 2-3x attention speedup
- Gradient checkpointing to reduce memory usage
- torch.compile() for 10-30% overall speedup
- Increased DataLoader parallelism (12 workers, prefetch_factor=4)
- Reduced GPU-CPU synchronization
- Uses Qwen3-4B-Instruct-2507 decoder (smaller, faster)

Usage:
    python train_unified_mapper.py
    python train_unified_mapper.py --resume Mapper_Checkpoints/unified_mapper_optimized.pth
"""

import os

# Set CUDA memory allocator to reduce fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import glob
import warnings
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from mapper import Mapper, LowRankMapper, MLPlMapper
from utils import is_valid_python
from model_config import DEFAULT_ENCODER, DEFAULT_DECODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder


# ============================================================================
# Task-Specific Skeleton Prompts
# ============================================================================

TASK_PROMPTS = {
    'tsp_construct': "Based on the above context, write a Python function that implements a TSP heuristic. Visit each node once and return to the start by choosing the next node step-by-step from the current node.",
    'cvrp_construct': "Based on the above context, write a Python function that implements a CVRP heuristic. Build routes from a depot while respecting vehicle capacity and choosing the next customer each step.",
    'vrptw_construct': "Based on the above context, write a Python function that implements a VRPTW heuristic. Choose the next customer step-by-step while respecting time windows, capacity, and travel times.",
    'jssp_construct': "Based on the above context, write a Python function that implements a JSSP heuristic. Schedule the next feasible operation to reduce overall makespan given machine and job constraints.",
    'knapsack_construct': "Based on the above context, write a Python function that implements a 0/1 knapsack heuristic. Select the next item to pack given remaining capacity and item weights/values.",
    'online_bin_packing': "Based on the above context, write a Python function that implements an online bin packing heuristic. Return a priority score for each bin to place the incoming item.",
    'qap_construct': "Based on the above context, write a Python function that implements a QAP heuristic. Extend a partial assignment of facilities to locations to reduce interaction cost.",
    'cflp_construct': "Based on the above context, write a Python function that implements a CFLP heuristic. Assign customers to facilities step-by-step while respecting capacity and minimizing cost.",
    'set_cover_construct': "Based on the above context, write a Python function that implements a greedy set cover heuristic. Pick the next subset to cover remaining elements.",
    'admissible_set': "Based on the above context, write a Python function that implements a heuristic priority score for adding a candidate vector to an admissible set."
}

# ============================================================================
# Prompt Augmentation: Problem-Class and General Prompts
# ============================================================================

PROBLEM_CLASS_PROMPTS = [
    "Based on the above context, write a Python function that implements a heuristic for a combinatorial optimization problem.",
    "Following the preceding context, implement a greedy heuristic function in Python for an optimization problem.",
    "Based on the above, write a Python function for solving a discrete optimization problem.",
    "Using the context provided above, create a Python heuristic for a routing or scheduling problem.",
    "Based on the preceding information, write a constructive heuristic function in Python.",
    "Following the above context, implement an optimization heuristic in Python.",
]

GENERAL_PROMPTS = [
    "Based on the above context, write a Python function.",
    "Following the preceding context, write a Python heuristic function.",
    "Based on the above, implement the described algorithmic approach in Python.",
    "Using the context provided above, write the corresponding function in Python.",
    "Based on the preceding information, create a Python function.",
    "Following the above context, implement this algorithm in Python.",
]


def sample_prompt_with_augmentation(
    task_name: str,
    task_specific_prob: float = 0.60,
    problem_class_prob: float = 0.20,
    general_prob: float = 0.20,
    rng: Optional[np.random.Generator] = None
) -> str:
    """Sample a prompt using augmentation strategy for robustness."""
    if rng is None:
        rng = np.random.default_rng()

    total = task_specific_prob + problem_class_prob + general_prob
    task_specific_prob /= total
    problem_class_prob /= total
    general_prob /= total

    choice = rng.choice(
        ['task_specific', 'problem_class', 'general'],
        p=[task_specific_prob, problem_class_prob, general_prob]
    )

    if choice == 'task_specific':
        return TASK_PROMPTS.get(task_name, PROBLEM_CLASS_PROMPTS[0])
    elif choice == 'problem_class':
        return rng.choice(PROBLEM_CLASS_PROMPTS)
    else:
        return rng.choice(GENERAL_PROMPTS)


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_all_augmented_heuristics(task_dir: str = "task") -> List[Tuple[str, str, str, str]]:
    """Load all augmented heuristics from all tasks."""
    all_heuristics = []
    augmented_files = glob.glob(os.path.join(task_dir, "*", "augmented.json"))

    print(f"\n{'='*70}")
    print(f"Loading Augmented Heuristics from All Tasks")
    print(f"{'='*70}")
    print(f"Found {len(augmented_files)} task directories\n")

    for aug_file in sorted(augmented_files):
        task_name = Path(aug_file).parent.name
        skeleton_prompt = TASK_PROMPTS.get(task_name)

        if skeleton_prompt is None:
            warnings.warn(f"No skeleton prompt defined for task '{task_name}', skipping...")
            continue

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
            all_heuristics.append((code, task_name, skeleton_prompt, heuristic_name))
            valid_count += 1

        print(f"  {task_name:25s}: {valid_count:3d} heuristics")

    print(f"\n{'='*70}")
    print(f"Total: {len(all_heuristics)} valid heuristics across {len(set(h[1] for h in all_heuristics))} tasks")
    print(f"{'='*70}\n")

    return all_heuristics


def encode_all_heuristics(
    heuristics_list: List[Tuple[str, str, str, str]],
    encoder_model: SentenceTransformer,
    device: str = "cuda",
    batch_size: int = 64  # Increased for faster encoding
) -> pd.DataFrame:
    """Encode all heuristics using the code encoder."""
    print(f"\n{'='*70}")
    print(f"Encoding All Heuristics")
    print(f"{'='*70}\n")

    codes = [h[0] for h in heuristics_list]
    tasks = [h[1] for h in heuristics_list]
    prompts = [h[2] for h in heuristics_list]
    names = [h[3] for h in heuristics_list]

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

    embeddings = np.vstack(embeddings)

    print(f"\n Encoded all programs")
    print(f"  Embedding shape: {embeddings.shape}")

    df = pd.DataFrame({
        'code': codes,
        'z': list(embeddings),
        'task': tasks,
        'skeleton_prompt': prompts,
        'heuristic_name': names
    })

    print(f"\nDataset Statistics:")
    print(f"{'='*70}")
    task_counts = df['task'].value_counts().sort_index()
    for task, count in task_counts.items():
        print(f"  {task:25s}: {count:3d} programs")
    print(f"{'='*70}\n")

    return df


# ============================================================================
# Checkpoint Loading
# ============================================================================

def load_checkpoint_for_resume(checkpoint_path: str, mapper_model, optimizer=None, scheduler=None, device=None):
    """Load checkpoint for resuming training."""
    print(f"\n{'='*70}")
    print(f"Loading Checkpoint for Resume Training")
    print(f"{'='*70}\n")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    mapper_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model state from: {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state tensors to the correct device
        if device is not None:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        print(f"Loaded optimizer state")

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded scheduler state")

    start_epoch = checkpoint.get('epoch', 0)
    print(f"\nCheckpoint Metadata:")
    print(f"  Epoch: {start_epoch}")
    print(f"  Total programs: {checkpoint.get('total_programs', 'N/A')}")
    print(f"{'='*70}\n")

    return start_epoch, checkpoint


# ============================================================================
# Custom Collate Function
# ============================================================================

def multi_task_collate_fn(batch):
    """Collate function that handles per-sample data with task names."""
    codes, zs, task_names = zip(*batch)
    zs = torch.tensor(np.stack(zs), dtype=torch.float32)
    return list(codes), zs, list(task_names)


# ============================================================================
# Optimized Training Function
# ============================================================================

def train_unified_mapper_optimized(
    df: pd.DataFrame,
    mapper_model,
    optimizer,
    decoder_model,
    decoder_tokenizer,
    batch_size: int = 16,           # Increased from 4
    epochs: int = 30,
    accumulation_steps: int = 1,    # Reduced from 2 (larger batch eliminates need)
    max_length: Optional[int] = 1024,
    verbose: bool = True,
    checkpoint_dir: str = "Mapper_Checkpoints",
    task_specific_prob: float = 0.60,
    problem_class_prob: float = 0.20,
    general_prob: float = 0.20,
    seed: int = 42,
    start_epoch: int = 0,
    scheduler=None
):
    """
    Optimized training for unified multi-task mapper.

    Optimizations:
    - Larger batch size (16) for better GPU utilization
    - No gradient accumulation needed with larger batches
    - Increased DataLoader workers and prefetching
    - Reduced GPU-CPU synchronization (loss accumulation on GPU)
    - torch.cuda.empty_cache() only at end of training
    """
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence

    print(f"\n{'='*70}")
    print(f"Optimized Training: Unified Multi-Task Mapper")
    print(f"{'='*70}\n")

    # Prepare dataset
    dataset = list(zip(
        df['code'].tolist(),
        df['z'].tolist(),
        df['task'].tolist()
    ))

    # Optimized DataLoader with more workers and prefetching
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multi_task_collate_fn,
        num_workers=12,             # Increased from 8 (using 12 of 16 vCPUs)
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,          # Prefetch more batches
        drop_last=True              # Consistent batch sizes for torch.compile
    )

    rng = np.random.default_rng(seed)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )

    # Setup models
    mapper_model.train()
    decoder_model.eval()
    for p in decoder_model.parameters():
        p.requires_grad = False
    decoder_model.config.use_cache = False

    embed_layer = decoder_model.get_input_embeddings()
    first_dev = embed_layer.weight.device
    embed_dtype = embed_layer.weight.dtype
    mapper_model = mapper_model.to(first_dev)

    pad_id = decoder_tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("decoder_tokenizer.pad_token_id must be set.")

    print(f"Training Configuration (Optimized):")
    print(f"  Batch size: {batch_size}")
    print(f"  Effective batch: {batch_size * accumulation_steps}")
    print(f"  Epochs: {start_epoch} -> {epochs}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Total programs: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataloader)}")
    print(f"  DataLoader workers: 12")
    print(f"  Prefetch factor: 4")
    print(f"\nPrompt Augmentation:")
    print(f"  Task-specific: {task_specific_prob*100:.0f}%")
    print(f"  Problem-class: {problem_class_prob*100:.0f}%")
    print(f"  General: {general_prob*100:.0f}%")
    if start_epoch > 0:
        print(f"\n  Resuming from epoch {start_epoch}")
    print(f"\nStarting training...\n")

    # Training loop
    for epoch in range(start_epoch, epochs):
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)
        # Accumulate loss on GPU to avoid CPU sync per batch
        running_loss = torch.tensor(0.0, device=first_dev)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for step, (code_batch, z_batch, task_batch) in enumerate(pbar):
            z_batch = z_batch.to(first_dev, non_blocking=True)
            B = len(code_batch)

            code_texts = [f"```python\n{code}\n```" for code in code_batch]

            prompt_id_tensors = []
            target_id_tensors = []

            for task_name, code_text in zip(task_batch, code_texts):
                skel_text = sample_prompt_with_augmentation(
                    task_name=task_name,
                    task_specific_prob=task_specific_prob,
                    problem_class_prob=problem_class_prob,
                    general_prob=general_prob,
                    rng=rng
                )

                prompt_messages = [{"role": "user", "content": skel_text}]
                full_messages = [
                    {"role": "user", "content": skel_text},
                    {"role": "assistant", "content": code_text},
                ]

                prompt_ids = decoder_tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=max_length,
                )
                full_ids = decoder_tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    truncation=True,
                    max_length=max_length,
                )

                if len(full_ids) < len(prompt_ids):
                    raise ValueError("chat template produced invalid prompt/full lengths.")

                target_ids = full_ids[len(prompt_ids):]
                prompt_id_tensors.append(torch.tensor(prompt_ids, dtype=torch.long))
                target_id_tensors.append(torch.tensor(target_ids, dtype=torch.long))

            prompt_ids = pad_sequence(prompt_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)
            target_ids = pad_sequence(target_id_tensors, batch_first=True, padding_value=pad_id).to(first_dev)

            prompt_embeds = embed_layer(prompt_ids)
            soft_prompt_embeds = mapper_model(z_batch).to(first_dev, dtype=embed_dtype)

            if soft_prompt_embeds.dim() != 3 or soft_prompt_embeds.size(0) != B:
                raise ValueError(f"mapper output must be [B, S_soft, D]; got {tuple(soft_prompt_embeds.shape)}")

            target_embeds = embed_layer(target_ids)
            inputs_embeds = torch.cat([soft_prompt_embeds, prompt_embeds, target_embeds], dim=1)

            ignore_left = torch.full(
                (B, prompt_embeds.size(1) + soft_prompt_embeds.size(1)),
                -100, dtype=torch.long, device=first_dev,
            )
            tgt_labels = target_ids.masked_fill(target_ids == pad_id, -100)
            labels = torch.cat([ignore_left, tgt_labels], dim=1)

            left_mask = torch.ones(B, prompt_embeds.size(1) + soft_prompt_embeds.size(1), dtype=torch.long, device=first_dev)
            right_mask = (target_ids != pad_id).long().to(first_dev)
            att_mask = torch.cat([left_mask, right_mask], dim=1)

            out = decoder_model(inputs_embeds=inputs_embeds, attention_mask=att_mask, labels=labels)
            loss = out.loss / accumulation_steps

            loss.backward()
            running_loss += loss.detach()
            total_steps += 1

            if total_steps % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Update progress bar with current loss (minimal sync)
            if step % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        if total_steps % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Single GPU-CPU sync per epoch
        avg_loss = running_loss.item() / max(1, len(dataloader))
        scheduler.step(avg_loss)

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"unified_mapper_optimized_epoch{epoch+1}.pth")
            checkpoint_data = {
                'model_state_dict': mapper_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
            }
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            # Note: mapper_type and internal_dim will be added in final checkpoint
            # These intermediate checkpoints are for resume only
            torch.save(checkpoint_data, checkpoint_path)
            if verbose:
                print(f"  Checkpoint saved: {checkpoint_path}")

    # Only clear cache at end of training
    torch.cuda.empty_cache()
    return mapper_model


# ============================================================================
# Main Training Script
# ============================================================================

def main(resume_checkpoint: Optional[str] = None, encoder_name: str = None, decoder_name: str = None, embedding_dim: int = None):
    """Main training pipeline with optimizations.

    Args:
        resume_checkpoint: Path to checkpoint file to resume training from.
        encoder_name: Encoder model name. Defaults to DEFAULT_ENCODER from model_config.py
        decoder_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
        embedding_dim: Matryoshka embedding dimension. If None, uses DEFAULT_MATRYOSHKA_DIM.
    """
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER
    if decoder_name is None:
        decoder_name = DEFAULT_DECODER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')
        print("TF32 matmul precision: enabled\n")

    # ========================================================================
    # Step 1: Load All Augmented Heuristics
    # ========================================================================

    heuristics_list = load_all_augmented_heuristics(task_dir="task")

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

    unified_df = encode_all_heuristics(
        heuristics_list=heuristics_list,
        encoder_model=encoder_model,
        device=device,
        batch_size=64  # Larger batch for encoding
    )

    # Free encoder memory
    del encoder_model
    torch.cuda.empty_cache()

    # ========================================================================
    # Step 4: Load Decoder Model (Qwen3-4B with Flash Attention 2)
    # ========================================================================

    print(f"{'='*70}")
    print(f"Loading Decoder Model ({decoder_name})")
    print(f"{'='*70}\n")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        decoder_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 2-3x attention speedup
        trust_remote_code=True
    )

    # Enable gradient checkpointing to reduce memory (recompute activations during backward)
    decoder_model.gradient_checkpointing_enable()

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        decoder_name,
        trust_remote_code=True
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    print("Decoder loaded with Flash Attention 2 + Gradient Checkpointing\n")

    # ========================================================================
    # Step 5: Initialize Mapper (LowRankMapper for parameter efficiency)
    # ========================================================================

    print(f"{'='*70}")
    print("Initializing Mapper (LowRankMapper)")
    print(f"{'='*70}\n")

    embeddings = np.stack(unified_df['z'].values)
    input_dim = embeddings.shape[1]
    output_dim = decoder_model.config.hidden_size
    num_tokens = 16

    # LowRankMapper hyperparameters
    internal_dim = 512       # Hidden dimension in shared MLP
    attn_heads = 2           # Number of attention heads
    attn_dropout = 0.1       # Attention dropout
    ffn_dropout = 0.1        # FFN and shared MLP dropout
    scale = 0.1              # Output scaling factor
    use_ffn = True           # Include FFN block after attention

    # Use LowRankMapper with attention for better parameter efficiency on smaller datasets
    # For ~4600 samples, LowRankMapper (~1.8M params) is safer than MLPlMapper (~40M+ params)
    mapper_model = LowRankMapper(
        input_dim=input_dim,
        output_dim=output_dim,
        num_tokens=num_tokens,
        internal_dim=internal_dim,
        attn_heads=attn_heads,
        attn_dropout=attn_dropout,
        ffn_dropout=ffn_dropout,
        scale=scale,
        use_ffn=use_ffn
    )
    mapper_type = 'LowRankMapper'

    # Note: torch.compile with "reduce-overhead" uses CUDA graphs which conflict
    # with gradient checkpointing. Use "default" mode instead for compatibility.
    mapper_model = torch.compile(mapper_model, mode="default")

    num_params = sum(p.numel() for p in mapper_model.parameters())
    print(f"Mapper Architecture ({mapper_type}):")
    print(f"  Input: {input_dim}D (code embeddings)")
    print(f"  Output: {num_tokens} tokens x {output_dim}D (soft prompts)")
    print(f"  Internal dim: {internal_dim}")
    print(f"  Attention heads: {attn_heads}")
    print(f"  Scale: {scale}")
    print(f"  Use FFN: {use_ffn}")
    print(f"  Parameters: {num_params:,}")
    print(f"  torch.compile: enabled (default mode)\n")

    # ========================================================================
    # Step 6: Setup Optimizer and Scheduler
    # ========================================================================

    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(mapper_model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # ========================================================================
    # Step 7: Load Checkpoint if Resuming
    # ========================================================================

    start_epoch = 0
    if resume_checkpoint is not None:
        # Get the device from the decoder's embedding layer (where mapper will be moved)
        embed_layer = decoder_model.get_input_embeddings()
        target_device = embed_layer.weight.device
        start_epoch, _ = load_checkpoint_for_resume(
            checkpoint_path=resume_checkpoint,
            mapper_model=mapper_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=target_device
        )

    # ========================================================================
    # Step 8: Create Checkpoint Directory
    # ========================================================================

    checkpoint_dir = "Mapper_Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ========================================================================
    # Step 9: Train Unified Mapper (Optimized)
    # ========================================================================

    trained_mapper = train_unified_mapper_optimized(
        df=unified_df,
        mapper_model=mapper_model,
        optimizer=optimizer,
        decoder_model=decoder_model,
        decoder_tokenizer=decoder_tokenizer,
        batch_size=8,               # Reduced to avoid OOM on logits tensor
        epochs=30,
        accumulation_steps=2,       # Effective batch = 16
        max_length=1024,            # Reduced to save memory
        verbose=True,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        scheduler=scheduler
    )

    # ========================================================================
    # Step 10: Save Final Checkpoint
    # ========================================================================

    print(f"\n{'='*70}")
    print("Saving Final Checkpoint")
    print(f"{'='*70}\n")

    task_counts = unified_df['task'].value_counts().to_dict()

    checkpoint_data = {
        'model_state_dict': trained_mapper.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_tokens': num_tokens,
        'internal_dim': internal_dim,
        'mapper_type': mapper_type,
        'attn_heads': attn_heads,
        'attn_dropout': attn_dropout,
        'ffn_dropout': ffn_dropout,
        'scale': scale,
        'use_ffn': use_ffn,
        'tasks_trained': list(TASK_PROMPTS.keys()),
        'total_programs': len(unified_df),
        'programs_per_task': task_counts,
        'epoch': 30,
        'encoder_model': encoder_name,
        'decoder_model': decoder_name,
        'optimizations': {
            'flash_attention_2': True,
            'gradient_checkpointing': True,
            'torch_compile': 'default',
            'batch_size': 8,
            'accumulation_steps': 2,
            'effective_batch_size': 16,
            'max_length': 1024,
            'num_workers': 12,
            'prefetch_factor': 4,
        },
        'prompt_augmentation': {
            'strategy': 'Strategy 1: Task-Specific + Problem-Class + General',
            'task_specific_prob': 0.60,
            'problem_class_prob': 0.20,
            'general_prob': 0.20,
        }
    }

    final_path = os.path.join(checkpoint_dir, "unified_mapper_optimized.pth")
    torch.save(checkpoint_data, final_path)

    # Save weights-only checkpoint (smaller, for inference/deployment)
    weights_only_data = {
        'model_state_dict': trained_mapper.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_tokens': num_tokens,
        'internal_dim': internal_dim,
        'mapper_type': mapper_type,
        'attn_heads': attn_heads,
        'attn_dropout': attn_dropout,
        'ffn_dropout': ffn_dropout,
        'scale': scale,
        'use_ffn': use_ffn,
        'tasks_trained': list(TASK_PROMPTS.keys()),
        'encoder_model': encoder_name,
        'decoder_model': decoder_name,
    }
    weights_only_path = os.path.join(checkpoint_dir, "unified_mapper_weights_only.pth")
    torch.save(weights_only_data, weights_only_path)

    print(f"Final model saved to: {final_path}")
    print(f"Weights-only model saved to: {weights_only_path}")
    print(f"\nTraining Complete!")
    print(f"{'='*70}\n")

    print("Training Summary:")
    print(f"  Total programs: {len(unified_df)}")
    print(f"  Tasks trained: {len(set(unified_df['task']))}")
    print(f"  Mapper type: {mapper_type}")
    print(f"  Model parameters: {num_params:,}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Decoder: {decoder_name}")
    print(f"  Optimizations: Flash Attention 2, Gradient Checkpointing, torch.compile(default)")
    print(f"  Batch: 8 x 2 accumulation = 16 effective")
    print(f"  Full checkpoint (for resume): {final_path}")
    print(f"  Weights-only (for inference): {weights_only_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized unified multi-task mapper training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=DEFAULT_ENCODER,
        help=f"Encoder model name (default: {DEFAULT_ENCODER})"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=DEFAULT_DECODER,
        help=f"Decoder model name (default: {DEFAULT_DECODER})"
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
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        embedding_dim=getattr(args, 'embedding_dim', None)
    )
