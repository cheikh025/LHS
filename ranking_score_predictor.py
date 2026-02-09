"""
Ranking-based score predictor for gradient-based optimization in prior space (u-space).

Uses pairwise ranking loss instead of MSE regression to handle small datasets.
From n programs, creates n*(n-1)/2 pairs for training.

Encoder: Qwen/Qwen3-Embedding-0.6B 
Pipeline: code -> z (encoder) -> u (flow) -> score (predictor)

IMPORTANT: Uses PROGRAM-LEVEL split to avoid validation leakage.
- Train/val split is done on PROGRAMS first, then pairs are created within each split
- This ensures validation pairs contain only programs the model has never seen
- Reports Spearman's rho, Kendall's tau, and held-out pair accuracy
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau

from model_config import DEFAULT_ENCODER, DEFAULT_MATRYOSHKA_DIM
from load_encoder_decoder import load_encoder


class PairwiseDataset(Dataset):
    """Dataset of (u_better, u_worse, score_diff) pairs for ranking."""

    def __init__(self, u_better: torch.Tensor, u_worse: torch.Tensor, score_diffs: torch.Tensor):
        self.u_better = u_better
        self.u_worse = u_worse
        self.score_diffs = score_diffs

    def __len__(self):
        return len(self.u_better)

    def __getitem__(self, idx):
        return self.u_better[idx], self.u_worse[idx], self.score_diffs[idx]


class RankingScorePredictor(nn.Module):
    """
    MLP that predicts score from prior-space vector u.
    Trained with ranking loss for better generalization on small datasets.

    Uses Qwen/Qwen3-Embedding-0.6B for z embeddings, then normalizing flow for z -> u.

    R: R^d -> R
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Predict score from prior-space vector."""
        return self.network(u)


def load_heuristics(task_name: str) -> Dict[str, str]:
    """Load heuristics from JSON file for a given task."""
    heuristics_path = Path(f"task/{task_name}/augmented.json")

    if not heuristics_path.exists():
        raise FileNotFoundError(f"Heuristics file not found: {heuristics_path}")

    with open(heuristics_path, 'r') as f:
        programs = json.load(f)

    return programs


def get_evaluator(task_name: str):
    """Get the evaluator class for a given task."""
    task_evaluators = {
        'tsp_construct': ('task.tsp_construct.evaluation', 'TSPEvaluation'),
        'cvrp_construct': ('task.cvrp_construct.evaluation', 'CVRPEvaluation'),
        'vrptw_construct': ('task.vrptw_construct.evaluation', 'VRPTWEvaluation'),
        'jssp_construct': ('task.jssp_construct.evaluation', 'JSSPEvaluation'),
        'knapsack_construct': ('task.knapsack_construct.evaluation', 'KnapsackEvaluation'),
        'online_bin_packing': ('task.online_bin_packing.evaluation', 'OBPEvaluation'),
        'qap_construct': ('task.qap_construct.evaluation', 'QAPEvaluation'),
        'set_cover_construct': ('task.set_cover_construct.evaluation', 'SCPEvaluation'),
        'cflp_construct': ('task.cflp_construct.evaluation', 'CFLPEvaluation'),
        'admissible_set': ('task.admissible_set.evaluation', 'ASPEvaluation'),
    }

    if task_name not in task_evaluators:
        raise ValueError(f"Unknown task: {task_name}")

    module_path, class_name = task_evaluators[task_name]
    module = __import__(module_path, fromlist=[class_name])
    EvaluatorClass = getattr(module, class_name)
    return EvaluatorClass()


def evaluate_programs(task_name: str, programs: Dict[str, str], use_secure: bool = True, num_workers: int = 4) -> pd.DataFrame:
    """
    Evaluate all programs for a task and return DataFrame with scores.

    Args:
        task_name: Name of the task
        programs: Dictionary of {name: code} pairs
        use_secure: Whether to use SecureEvaluator wrapper
        num_workers: Number of parallel workers for evaluation

    Returns:
        DataFrame with columns ['name', 'code', 'score']
    """
    evaluator = get_evaluator(task_name)

    if use_secure:
        from base.evaluate import SecureEvaluator
        evaluator = SecureEvaluator(evaluator, debug_mode=False)

    def eval_single(name_code):
        """Evaluate a single program."""
        name, code = name_code
        try:
            score = evaluator.evaluate_program(code)
            if score is not None and np.isfinite(score):
                return {'name': name, 'code': code, 'score': score}
        except Exception:
            pass
        return None

    results = []

    # Parallel evaluation
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_single, (name, code)): name
                   for name, code in programs.items()}

        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Evaluating {task_name} ({num_workers} workers)"):
            result = future.result()
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)


def get_encoder_model(device: str = 'cuda', model_name: str = None, truncate_dim: int = None):
    """
    Load the same encoder model used in unified training and programDB.

    Args:
        device: Device to load the model on.
        model_name: Encoder model name. Defaults to DEFAULT_ENCODER from model_config.py
        truncate_dim: Matryoshka dimension. If None, uses DEFAULT_MATRYOSHKA_DIM from config.

    Returns:
        Tuple of (encoder_model, embedding_dim)
    """
    return load_encoder(model_name=model_name, device=device, truncate_dim=truncate_dim)


def encode_programs(codes: List[str], encoder_model=None, device: str = 'cuda', batch_size: int = 32) -> torch.Tensor:
    """
    Encode programs using Qwen/Qwen3-Embedding-0.6B SentenceTransformer.

    Args:
        codes: List of code strings
        encoder_model: Optional pre-loaded encoder model. If None, will load the default encoder.
        device: Device for encoding
        batch_size: Batch size for encoding

    Returns:
        Tensor of embeddings [n, embedding_dim]
    """
    # Load encoder if not provided
    if encoder_model is None:
        encoder_model, _ = get_encoder_model(device)

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(codes), batch_size), desc="Encoding programs"):
            batch_codes = codes[i:i+batch_size]
            batch_embeddings = encoder_model.encode(
                batch_codes,
                convert_to_tensor=True,
                device=device,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu().numpy())

    return torch.tensor(np.vstack(embeddings), dtype=torch.float32, device=device)


def create_pairs_from_programs(
    scores: np.ndarray,
    u_embeddings: torch.Tensor,
    program_indices: List[int],
    min_score_diff: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create pairwise ranking data from a subset of programs.

    Args:
        scores: Array of ALL program scores
        u_embeddings: Tensor of ALL program u-space embeddings [n, dim]
        program_indices: Indices of programs to use for pair creation
        min_score_diff: Minimum score difference to exclude ties

    Returns:
        u_better: U-space vectors of better programs
        u_worse: U-space vectors of worse programs
        score_diffs: Score differences (always positive)
    """
    device = u_embeddings.device

    better_indices = []
    worse_indices = []
    diffs = []

    # Only create pairs from the specified program indices
    for i, j in combinations(program_indices, 2):
        diff = scores[i] - scores[j]

        if abs(diff) <= min_score_diff:
            continue

        if diff > 0:
            better_indices.append(i)
            worse_indices.append(j)
            diffs.append(diff)
        else:
            better_indices.append(j)
            worse_indices.append(i)
            diffs.append(-diff)

    if len(better_indices) == 0:
        return None, None, None

    u_better = u_embeddings[better_indices]
    u_worse = u_embeddings[worse_indices]
    score_diffs = torch.tensor(diffs, dtype=torch.float32, device=device)

    return u_better, u_worse, score_diffs


def create_dataset_from_task(
    task_name: str,
    flow_model,
    min_score_diff: float = 0.0,
    device: str = 'cuda',
    encoder_model=None,
    num_workers: int = 4,
    val_split: float = 0.2
) -> Tuple[PairwiseDataset, PairwiseDataset, pd.DataFrame, torch.Tensor, List[int], List[int]]:
    """
    Create pairwise datasets from a task's heuristics with PROGRAM-LEVEL split.

    Args:
        task_name: Name of the task
        flow_model: Trained normalizing flow
        min_score_diff: Minimum score difference for pairs
        device: Device to use
        encoder_model: Optional pre-loaded encoder model
        num_workers: Number of parallel workers for evaluation
        val_split: Fraction of PROGRAMS (not pairs) for validation

    Returns:
        train_dataset: PairwiseDataset for training (pairs from train programs only)
        val_dataset: PairwiseDataset for validation (pairs from val programs only)
        df: DataFrame with evaluated programs
        u_embeddings: All u-space embeddings
        train_indices: Indices of programs used for training
        val_indices: Indices of programs used for validation
    """
    # Load and evaluate
    programs = load_heuristics(task_name)
    print(f"Loaded {len(programs)} programs from {task_name}")

    df = evaluate_programs(task_name, programs, num_workers=num_workers)
    print(f"Successfully evaluated {len(df)} programs")

    if len(df) < 4:
        raise ValueError(f"Need at least 4 programs for train/val split, got {len(df)}")

    # Encode programs (z-space)
    codes = df['code'].tolist()
    z_embeddings = encode_programs(codes, encoder_model=encoder_model, device=device)

    # Store z embeddings in dataframe
    df['z'] = list(z_embeddings.cpu().numpy())

    # Map z -> u using flow
    flow_model.eval()
    with torch.no_grad():
        u_embeddings, _ = flow_model(z_embeddings.to(device))

    # Store u embeddings in dataframe
    df['u'] = list(u_embeddings.cpu().numpy())

    scores = df['score'].values

    # ========================================
    # PROGRAM-LEVEL SPLIT 
    # ========================================
    n_programs = len(df)
    n_val = max(2, int(n_programs * val_split))  # At least 2 programs for validation
    n_train = n_programs - n_val

    # Random shuffle of program indices
    all_indices = np.random.permutation(n_programs).tolist()
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    print(f"\n*** PROGRAM-LEVEL SPLIT (no leakage) ***")
    print(f"  Train programs: {n_train}")
    print(f"  Val programs: {n_val}")

    # Create pairs ONLY from train programs
    u_better_train, u_worse_train, diffs_train = create_pairs_from_programs(
        scores, u_embeddings, train_indices, min_score_diff
    )

    # Create pairs ONLY from val programs
    u_better_val, u_worse_val, diffs_val = create_pairs_from_programs(
        scores, u_embeddings, val_indices, min_score_diff
    )

    if u_better_train is None or len(u_better_train) == 0:
        raise ValueError("No valid training pairs created")

    if u_better_val is None or len(u_better_val) == 0:
        raise ValueError("No valid validation pairs created - try increasing val_split or dataset size")

    n_train_pairs = len(u_better_train)
    n_val_pairs = len(u_better_val)

    print(f"  Train pairs: {n_train_pairs} (from {n_train} programs)")
    print(f"  Val pairs: {n_val_pairs} (from {n_val} programs)")
    print(f"  Train amplification: {n_train_pairs / n_train:.1f}x")
    print(f"  Val amplification: {n_val_pairs / n_val:.1f}x")

    train_dataset = PairwiseDataset(u_better_train, u_worse_train, diffs_train)
    val_dataset = PairwiseDataset(u_better_val, u_worse_val, diffs_val)

    return train_dataset, val_dataset, df, u_embeddings, train_indices, val_indices


class RankingLoss(nn.Module):
    """
    Soft ranking loss using sigmoid + BCE.
    Provides smooth gradients for optimization.

    P(i > j) = sigmoid(tau * (R(u_i) - R(u_j)))
    loss = -log(P(i > j))

    Uses numerically stable logsigmoid implementation.
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(self, score_better: torch.Tensor, score_worse: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score_better: Predicted scores for better programs [batch]
            score_worse: Predicted scores for worse programs [batch]

        Returns:
            loss: Scalar loss value
        """
        diff = score_better - score_worse
        # Use numerically stable logsigmoid instead of log(sigmoid())
        loss = -F.logsigmoid(self.tau * diff).mean()
        return loss


def compute_ranking_metrics(
    predictor: RankingScorePredictor,
    u_embeddings: torch.Tensor,
    actual_scores: np.ndarray,
    program_indices: List[int],
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute ranking correlation metrics on a set of programs.

    Args:
        predictor: Trained ranking predictor
        u_embeddings: All program u-space embeddings
        actual_scores: All actual scores
        program_indices: Indices of programs to evaluate on
        device: Device to use

    Returns:
        Dictionary with spearman_rho, kendall_tau, pair_accuracy
    """
    predictor.eval()

    # Get u embeddings and scores for the specified programs
    u_subset = u_embeddings[program_indices].to(device)
    actual_subset = actual_scores[program_indices]

    # Predict scores
    with torch.no_grad():
        predicted = predictor(u_subset).squeeze(-1).cpu().numpy()  # Only squeeze last dim

    # Spearman's rank correlation
    spearman_rho, spearman_p = spearmanr(actual_subset, predicted)

    # Kendall's tau
    kendall_tau_val, kendall_p = kendalltau(actual_subset, predicted)

    # Pairwise accuracy
    correct = 0
    total = 0
    for i, j in combinations(range(len(program_indices)), 2):
        actual_diff = actual_subset[i] - actual_subset[j]
        pred_diff = predicted[i] - predicted[j]

        if actual_diff != 0:  # Skip ties
            if (actual_diff > 0 and pred_diff > 0) or (actual_diff < 0 and pred_diff < 0):
                correct += 1
            total += 1

    pair_accuracy = correct / total if total > 0 else 0.0

    return {
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau_val,
        'kendall_p': kendall_p,
        'pair_accuracy': pair_accuracy,
        'n_programs': len(program_indices),
        'n_pairs': total
    }


def train_ranking_predictor(
    predictor: RankingScorePredictor,
    train_dataset: PairwiseDataset,
    val_dataset: PairwiseDataset,
    u_embeddings: torch.Tensor,
    actual_scores: np.ndarray,
    train_indices: List[int],
    val_indices: List[int],
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    tau: float = 1.0,
    device: str = 'cuda',
    verbose: bool = True,
    patience: int = 20
) -> Tuple[RankingScorePredictor, Dict]:
    """
    Train ranking score predictor with PROGRAM-LEVEL validation.

    Args:
        predictor: RankingScorePredictor model
        train_dataset: PairwiseDataset from train programs only
        val_dataset: PairwiseDataset from val programs only
        u_embeddings: All program u-space embeddings (for computing metrics)
        actual_scores: All actual scores (for computing metrics)
        train_indices: Indices of training programs
        val_indices: Indices of validation programs
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        tau: Temperature for soft ranking loss
        device: Device to train on
        verbose: Print progress
        patience: Early stopping patience

    Returns:
        Trained predictor and training history
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n*** Training with PROGRAM-LEVEL split ***")
    print(f"Training on {len(train_dataset)} pairs from {len(train_indices)} programs")
    print(f"Validating on {len(val_dataset)} pairs from {len(val_indices)} programs")
    print(f"(Validation programs are COMPLETELY UNSEEN during training)\n")

    # Loss function
    criterion = RankingLoss(tau=tau)

    # Move to device
    predictor = predictor.to(device)
    predictor.train()

    # Optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_pair_acc': [], 'val_pair_acc': [],
        'val_spearman': [], 'val_kendall': []
    }
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        predictor.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for u_better, u_worse, _ in train_loader:
            u_better = u_better.to(device)
            u_worse = u_worse.to(device)

            optimizer.zero_grad()

            score_better = predictor(u_better).squeeze(-1)
            score_worse = predictor(u_worse).squeeze(-1)

            loss = criterion(score_better, score_worse)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_correct += (score_better > score_worse).sum().item()
            train_total += score_better.numel()

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_pair_acc = train_correct / max(train_total, 1)

        # Validation (on COMPLETELY UNSEEN programs)
        predictor.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for u_better, u_worse, _ in val_loader:
                u_better = u_better.to(device)
                u_worse = u_worse.to(device)

                score_better = predictor(u_better).squeeze(-1)
                score_worse = predictor(u_worse).squeeze(-1)

                loss = criterion(score_better, score_worse)
                val_loss += loss.item()

                val_correct += (score_better > score_worse).sum().item()
                val_total += score_better.numel()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_pair_acc = val_correct / max(val_total, 1)

        # Compute ranking metrics on held-out programs
        val_metrics = compute_ranking_metrics(
            predictor, u_embeddings, actual_scores, val_indices, device
        )

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_pair_acc'].append(train_pair_acc)
        history['val_pair_acc'].append(val_pair_acc)
        history['val_spearman'].append(val_metrics['spearman_rho'])
        history['val_kendall'].append(val_metrics['kendall_tau'])

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = predictor.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Pair Acc: {train_pair_acc:.2%}")
            print(f"  Val Loss:   {avg_val_loss:.4f} | Pair Acc: {val_pair_acc:.2%} (on UNSEEN programs)")
            print(f"  Val Spearman rho: {val_metrics['spearman_rho']:.4f} | Kendall tau: {val_metrics['kendall_tau']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            print()

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        predictor.load_state_dict(best_state)

    # Final metrics on held-out programs
    final_train_metrics = compute_ranking_metrics(
        predictor, u_embeddings, actual_scores, train_indices, device
    )
    final_val_metrics = compute_ranking_metrics(
        predictor, u_embeddings, actual_scores, val_indices, device
    )

    print(f"\n{'='*60}")
    print("FINAL RESULTS (Program-Level Split)")
    print(f"{'='*60}")
    print(f"\nTraining Programs ({len(train_indices)} programs, {final_train_metrics['n_pairs']} pairs):")
    print(f"  Pair Accuracy:  {final_train_metrics['pair_accuracy']:.2%}")
    print(f"  Spearman rho:   {final_train_metrics['spearman_rho']:.4f} (p={final_train_metrics['spearman_p']:.4f})")
    print(f"  Kendall tau:    {final_train_metrics['kendall_tau']:.4f} (p={final_train_metrics['kendall_p']:.4f})")

    print(f"\nHeld-Out Programs ({len(val_indices)} programs, {final_val_metrics['n_pairs']} pairs):")
    print(f"  Pair Accuracy:  {final_val_metrics['pair_accuracy']:.2%}")
    print(f"  Spearman rho:   {final_val_metrics['spearman_rho']:.4f} (p={final_val_metrics['spearman_p']:.4f})")
    print(f"  Kendall tau:    {final_val_metrics['kendall_tau']:.4f} (p={final_val_metrics['kendall_p']:.4f})")

    print(f"\n INTERPRETATION ")
    if final_val_metrics['pair_accuracy'] > 0.7:
        print("  Held-out pair accuracy > 70%: Predictor generalizes well!")
    elif final_val_metrics['pair_accuracy'] > 0.6:
        print("  Held-out pair accuracy 60-70%: Moderate generalization")
    else:
        print("  Held-out pair accuracy < 60%: POOR generalization - predictor may be memorizing!")
        print("  Gradient-based search may follow misleading gradients in unseen regions.")

    history['final_train_metrics'] = final_train_metrics
    history['final_val_metrics'] = final_val_metrics

    return predictor, history


def save_ranking_predictor(
    predictor: RankingScorePredictor,
    path: str,
    history: Optional[Dict] = None,
    extra_info: Optional[Dict] = None,
    encoder_name: str = None
):
    """Save ranking predictor to disk.

    Args:
        predictor: The ranking predictor model.
        path: Path to save the checkpoint.
        history: Optional training history.
        extra_info: Optional extra info to save.
        encoder_name: Encoder model name used. Defaults to DEFAULT_ENCODER.
    """
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER

    checkpoint = {
        'model_state_dict': predictor.state_dict(),
        'input_dim': predictor.input_dim,
        'hidden_dim': predictor.hidden_dim,
        'num_layers': predictor.num_layers,
        'space': 'u', 
        'encoder': encoder_name,  
    }

    if history is not None:
        checkpoint['history'] = history

    if extra_info is not None:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)
    print(f"Saved ranking predictor to {path}")


def load_ranking_predictor(path: str, device: str = 'cuda') -> Tuple[RankingScorePredictor, Dict]:
    """Load ranking predictor from disk."""
    checkpoint = torch.load(path, map_location=device)

    predictor = RankingScorePredictor(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint.get('num_layers', 2)
    )
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.to(device)

    print(f"Loaded ranking predictor from {path}")
    print(f"  Space: {checkpoint.get('space', 'u')}")
    print(f"  Encoder: {checkpoint.get('encoder', 'BAAI/bge-code-v1')}")
    print(f"  Input dim: {checkpoint['input_dim']}")

    return predictor, checkpoint


# ============================================================================
# Main training script
# ============================================================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Train ranking score predictor (u-space) with program-level split')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--flow_path', type=str, default='Flow_Checkpoints/unified_flow_final.pth', help='Path to trained flow model')
    parser.add_argument('--output_dir', type=str, default='Predictor_Checkpoints', help='Output directory for saved models')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: ranking_predictor_u_{task}.pth)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature for soft ranking loss (higher = sharper gradients)')
    parser.add_argument('--min_score_diff', type=float, default=0.0, help='Min score diff for pairs')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of PROGRAMS for validation (not pairs!)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for evaluation')
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER, help=f'Encoder model (default: {DEFAULT_ENCODER})')
    parser.add_argument('--embedding-dim', type=int, default=None,
                        help=f'Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or "model native"})')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default output filename if not provided
    if args.output is None:
        args.output = f"ranking_predictor_u_{args.task}.pth"

    # Full output path
    output_path = os.path.join(args.output_dir, args.output)

    print("="*70)
    print("Ranking Score Predictor Training (U-Space)")
    print("*** With PROGRAM-LEVEL Split (No Validation Leakage) ***")
    print("="*70)
    print(f"Task: {args.task}")
    print(f"Flow model: {args.flow_path}")
    print(f"Output: {output_path}")
    print(f"Loss: Soft ranking (tau={args.tau})")
    print(f"Val split: {args.val_split:.0%} of PROGRAMS")
    print(f"Device: {args.device}")
    print(f"Parallel workers: {args.num_workers}")
    print()

    # Load encoder model
    print(f"Loading encoder model ({args.encoder})...")
    encoder_model, embedding_dim = get_encoder_model(args.device, args.encoder, getattr(args, 'embedding_dim', None))
    print(f"Encoder loaded. Embedding dimension: {embedding_dim}\n")

    # Load flow model
    from normalizing_flow import NormalizingFlow

    flow_checkpoint = torch.load(args.flow_path, map_location=args.device)
    flow_dim = flow_checkpoint.get('dim', flow_checkpoint.get('flow_dim', flow_checkpoint.get('embedding_dim', 768)))
    flow_num_layers = flow_checkpoint.get('num_layers', flow_checkpoint.get('flow_num_layers', 4))
    flow_hidden_dim = flow_checkpoint.get('hidden_dim', flow_checkpoint.get('flow_hidden_dim', 128))
    flow_dropout = flow_checkpoint.get('dropout', flow_checkpoint.get('flow_dropout', 0.0))

    flow_model = NormalizingFlow(
        dim=flow_dim,
        num_layers=flow_num_layers,
        hidden_dim=flow_hidden_dim,
        dropout=flow_dropout
    )

    flow_state = flow_checkpoint.get('model_state_dict', flow_checkpoint.get('flow_state_dict'))
    if flow_state is None:
        raise KeyError("Flow checkpoint missing model_state_dict/flow_state_dict")

    flow_model.load_state_dict(flow_state)
    flow_model.to(args.device)
    flow_model.eval()
    print(f"Loaded flow model: dim={flow_dim}, layers={flow_num_layers}, hidden={flow_hidden_dim}, dropout={flow_dropout}")

    
    train_dataset, val_dataset, df, u_embeddings, train_indices, val_indices = create_dataset_from_task(
        task_name=args.task,
        flow_model=flow_model,
        min_score_diff=args.min_score_diff,
        device=args.device,
        encoder_model=encoder_model,
        num_workers=args.num_workers,
        val_split=args.val_split
    )

    # Free encoder memory after encoding
    del encoder_model
    torch.cuda.empty_cache()

    actual_scores = df['score'].values

    # Create predictor - input_dim comes from flow (u-space has same dim as z-space)
    input_dim = flow_dim
    print(f"\nEmbedding dimension: {input_dim}")
    predictor = RankingScorePredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    print(f"Created predictor: {sum(p.numel() for p in predictor.parameters()):,} parameters")

    # Train with program-level split
    predictor, history = train_ranking_predictor(
        predictor=predictor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        u_embeddings=u_embeddings,
        actual_scores=actual_scores,
        train_indices=train_indices,
        val_indices=val_indices,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        tau=args.tau,
        device=args.device,
        verbose=True
    )

    # Save
    save_ranking_predictor(
        predictor=predictor,
        path=output_path,
        history=history,
        extra_info={
            'task': args.task,
            'loss_type': 'soft',
            'tau': args.tau,
            'n_train_pairs': len(train_dataset),
            'n_val_pairs': len(val_dataset),
            'n_train_programs': len(train_indices),
            'n_val_programs': len(val_indices),
            'n_programs': len(df),
            'val_split_type': 'program_level'
        },
        encoder_name=args.encoder
    )

    print(f"\nDone! Model saved to {output_path}")


if __name__ == '__main__':
    main()
