# Latent Heuristic Search (LHS)

Implementation for the paper:
**"Latent Heuristic Search: Continuous Optimization for Automated Algorithm Design"**.

![Latent Heuristic Search Overview](Asset/LHS_v1.png)

## Overview

This repository implements a latent-space pipeline for automated heuristic design:

1. Encode heuristic programs into latent vectors (`z`).
2. Train a unified normalizing flow to map `z <-> u` (prior space).
3. Train a ranking predictor in `u`-space.
4. Optimize in continuous `u`-space with gradient ascent.
5. Decode optimized latents back into executable heuristic code and evaluate.

The framework supports multiple combinatorial optimization tasks under `task/`.

## Repository Structure

- `train_unified_flow.py`: trains one normalizing flow across all tasks.
- `train_unified_mapper.py`: trains the latent-to-soft-prompt mapper.
- `ranking_score_predictor.py`: trains task-specific `u`-space ranking predictor.
- `gradient_search.py`: main search loop (gradient ascent in prior space + generation + evaluation).
- `normalizing_flow.py`: RealNVP-style flow implementation.
- `mapper.py`: mapper architectures (`MLPlMapper`, `LowRankMapper`).
- `load_encoder_decoder.py`: model loading helpers (encoder/decoder).
- `model_config.py`: default model names and embedding settings.
- `utils.py`: shared code utilities.
- `base/`: core execution/evaluation framework (from LLM4AD base components).
- `task/`: task-specific datasets, templates, and evaluators.

## Supported Tasks

- `tsp_construct`
- `cvrp_construct`
- `vrptw_construct`
- `jssp_construct`
- `knapsack_construct`
- `online_bin_packing`
- `qap_construct`
- `cflp_construct`
- `set_cover_construct`
- `admissible_set`

## Environment

Recommended:

- Python 3.10+
- CUDA GPU
- PyTorch with CUDA

Install common dependencies:

```bash
pip install torch transformers sentence-transformers accelerate flash-attn numpy pandas scipy tqdm matplotlib
```

Notes:

- Decoder loading uses Flash Attention 2 (`attn_implementation="flash_attention_2"`).
- Update environment/model stack if your hardware or CUDA version differs.

## Default Models

From `model_config.py`:

- Encoder: `Qwen/Qwen3-Embedding-0.6B`
- Decoder: `Qwen/Qwen3-4B-Instruct-2507`
- Matryoshka embedding dim: `128`

## Training and Search Pipeline

Run from repository root.

### 1) Train Unified Flow

```bash
python3 train_unified_flow.py \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --embedding-dim 128
```

### 2) Train Unified Mapper

```bash
python3 train_unified_mapper.py \
  --encoder Qwen/Qwen3-Embedding-0.6B \
  --decoder Qwen/Qwen3-4B-Instruct-2507 \
  --embedding-dim 128
```

### 3) Train Task Ranking Predictor (u-space)

Example for TSP:

```bash
python3 ranking_score_predictor.py \
  --task tsp_construct \
  --flow_path Flow_Checkpoints/unified_flow_final.pth \
  --output_dir Predictor_Checkpoints
```

### 4) Run Gradient Search

```bash
python3 gradient_search.py \
  --task tsp_construct \
  --predictor Predictor_Checkpoints/ranking_predictor_u_tsp_construct.pth \
  --flow Flow_Checkpoints/unified_flow_final.pth \
  --mapper Mapper_Checkpoints/unified_mapper_optimized.pth \
  --output_dir gradient_search_u_results
```

Optional:

- `--llm_init --llm_init_count 20` to initialize search with freshly generated programs.

## More Parameters

The commands above show only the core arguments. Additional hyperparameters and runtime options are available in:

- `train_unified_flow.py`
- `train_unified_mapper.py`
- `ranking_score_predictor.py`
- `gradient_search.py`
- `model_config.py` (default model names and embedding dimension)

You can inspect all CLI options with:

```bash
python3 train_unified_flow.py --help
python3 train_unified_mapper.py --help
python3 ranking_score_predictor.py --help
python3 gradient_search.py --help
```

## Outputs

Typical generated directories/files:

- `Flow_Checkpoints/`
- `Mapper_Checkpoints/`
- `Predictor_Checkpoints/`
- `gradient_search_u_results/`

Search exports include discovered programs, per-program scores, and full program database snapshots.

## Acknowledgment

Parts of the evaluation/runtime infrastructure in `base/` and `task/` are adapted from the LLM4AD ecosystem: https://github.com/Optima-CityU/LLM4AD/tree/main
