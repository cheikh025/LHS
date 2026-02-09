"""
Gradient-based search in u-space (prior space) using trained ranking predictor and normalizing flow.

Pipeline:
1. Load trained ranking predictor R(u) and normalizing flow F
2. Gradient ascent in u-space to find u* = argmax [R(u) - λ∥u - u₀∥²]
3. Map u* → z* using flow inverse F^(-1)(u)
4. Generate code using mapper + decoder
5. Evaluate and save successful programs

Advantages over z-space search:
- Direct optimization in normalized prior space with predictor R(u) trained on u-vectors
- Flow inverse F^(-1)(u) naturally constrains search to the learned manifold
- Less risk of drifting to invalid embedding regions
- Optional trust region penalty λ∥u - u₀∥² for additional control

Uses same models as train_unified_mapper_optimized.py and train_unified_flow.py:
- Encoder: BAAI/bge-code-v1
- Decoder: Qwen/Qwen3-4B-Instruct-2507 (with Flash Attention 2)
- Flow: Unified normalizing flow trained on all tasks
- Predictor: U-space ranking predictor trained on prior-space vectors
"""

import os
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from mapper import Mapper
from normalizing_flow import NormalizingFlow
from model_config import DEFAULT_ENCODER, DEFAULT_DECODER, DEFAULT_MATRYOSHKA_DIM
from ranking_score_predictor import (
    RankingScorePredictor,
    load_ranking_predictor,
    get_encoder_model,
    load_heuristics,
    get_evaluator
)
from utils import extract_python_code_robust, is_valid_python
from base.code import TextFunctionProgramConverter
from base.evaluate import SecureEvaluator


# ============================================================================
# Task Templates for LLM Initialization
# ============================================================================

def load_task_template(task_name: str):
    """Load task template and description for LLM initialization."""
    import importlib.util
    import os

    template_path = os.path.join("task", task_name, "template.py")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    spec = importlib.util.spec_from_file_location("template", template_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, 'template_program', None), getattr(module, 'task_description', None)


def create_init_prompt(task_description: str, template_program: str) -> str:
    """Create prompt for LLM initialization (similar to EOH i1 prompt)."""
    # Extract function signature from template
    import re
    func_match = re.search(r'def\s+\w+\([^)]*\)[^:]*:', template_program)
    if func_match:
        func_signature = template_program[func_match.start():].split('"""')[0] + '"""'
        # Get docstring if present
        docstring_match = re.search(r'"""[\s\S]*?"""', template_program[func_match.start():])
        if docstring_match:
            func_signature = template_program[func_match.start():func_match.start() + docstring_match.end()]
    else:
        func_signature = template_program

    prompt = f"""{task_description}

Design a novel algorithm to solve this problem. Implement the following Python function:

{func_signature}

Provide only the complete Python function implementation, no explanations."""

    return prompt


# ============================================================================
# Task-Specific Prompts
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
# Model Loading
# ============================================================================

def load_decoder(model_name: str = None, device: str = "cuda"):
    """Load decoder model and tokenizer.

    Args:
        model_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
        device: Device to use.
    """
    if model_name is None:
        model_name = DEFAULT_DECODER
    print(f"Loading decoder: {model_name}...")

    decoder_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    print(f"  ✓ Decoder loaded with Flash Attention 2.")
    return decoder_model, decoder_tokenizer


def load_mapper(mapper_path: str, decoder_model, device: str = "cuda"):
    """Load trained mapper model (supports both OriginalMapper and LowRankMapper)."""
    from mapper import OriginalMapper, LowRankMapper

    print(f"Loading mapper from {mapper_path}...")

    checkpoint = torch.load(mapper_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        input_dim = checkpoint.get('input_dim')
        output_dim = checkpoint.get('output_dim')
        num_tokens = checkpoint.get('num_tokens')
        internal_dim = checkpoint.get('internal_dim')
        mapper_type = checkpoint.get('mapper_type', None)
        # Additional LowRankMapper parameters
        attn_heads = checkpoint.get('attn_heads', 2)
        attn_dropout = checkpoint.get('attn_dropout', 0.1)
        ffn_dropout = checkpoint.get('ffn_dropout', 0.1)
        scale = checkpoint.get('scale', 0.1)
        use_ffn = checkpoint.get('use_ffn', True)
    else:
        state_dict = checkpoint
        input_dim = None
        output_dim = None
        num_tokens = None
        internal_dim = None
        mapper_type = None
        attn_heads = 2
        attn_dropout = 0.1
        ffn_dropout = 0.1
        scale = 0.1
        use_ffn = True

    # Handle torch.compile() prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, removing '_orig_mod.' prefix...")
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Detect mapper type from state_dict keys if not specified
    if mapper_type is None:
        if 'feature_expander.weight' in state_dict:
            mapper_type = 'LowRankMapper'
        elif 'mlp.0.weight' in state_dict:
            mapper_type = 'OriginalMapper'
        else:
            raise ValueError("Cannot detect mapper type from checkpoint keys")

    print(f"  Mapper type: {mapper_type}")

    # Infer dimensions based on mapper type
    if mapper_type == 'LowRankMapper':
        if input_dim is None:
            input_dim = state_dict['feature_expander.weight'].shape[1]
        if output_dim is None:
            # Handle both old (index 2) and new (index 3 with dropout) shared_mlp
            if 'shared_mlp.3.weight' in state_dict:
                output_dim = state_dict['shared_mlp.3.weight'].shape[0]
            else:
                output_dim = state_dict['shared_mlp.2.weight'].shape[0]
        if num_tokens is None:
            num_tokens = state_dict['pos_embed'].shape[1]
        if internal_dim is None:
            internal_dim = state_dict['shared_mlp.0.weight'].shape[0]

        # Detect if attention-based (has ln_attn)
        has_attention = 'ln_attn.weight' in state_dict
        if has_attention:
            # Infer attn_heads from attention weight shape
            attn_embed_dim = state_dict['attn.in_proj_weight'].shape[1]
            # use_ffn detection
            use_ffn = 'ln_ffn.weight' in state_dict

        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num tokens: {num_tokens}")
        print(f"  Internal dim: {internal_dim}")
        if has_attention:
            print(f"  Has attention: True")
            print(f"  Use FFN: {use_ffn}")

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

    else:  # OriginalMapper
        if input_dim is None:
            input_dim = state_dict['mlp.0.weight'].shape[1]
        if output_dim is None:
            output_dim = decoder_model.config.hidden_size
        if num_tokens is None:
            final_output_size = state_dict['mlp.4.weight'].shape[0]
            num_tokens = final_output_size // output_dim

        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Num tokens: {num_tokens}")

        mapper_model = OriginalMapper(input_dim, output_dim, num_tokens)

    mapper_model.load_state_dict(state_dict)
    mapper_model.eval()

    return mapper_model


def load_flow(flow_path: str, device: str = "cuda"):
    """Load trained normalizing flow."""
    print(f"Loading normalizing flow from {flow_path}...")

    checkpoint = torch.load(flow_path, map_location=device)

    # Extract architecture parameters
    dim = checkpoint.get('dim', checkpoint.get('embedding_dim', 768))
    num_layers = checkpoint.get('num_layers', 4)  # Default to 4 for small datasets
    hidden_dim = checkpoint.get('hidden_dim', 128)  # Default to 128 for small datasets
    dropout = checkpoint.get('dropout', 0.0)  # Dropout will be disabled in eval mode

    print(f"  Dimension: {dim}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Dropout: {dropout} (disabled in eval mode)")

    # Create flow model with dropout (will be disabled in eval mode)
    flow_model = NormalizingFlow(dim=dim, num_layers=num_layers, hidden_dim=hidden_dim, dropout=dropout)
    flow_model.load_state_dict(checkpoint['model_state_dict'])
    flow_model.to(device)
    flow_model.eval()  # This disables dropout

    print(f"  ✓ Flow loaded")
    return flow_model


# ============================================================================
# Gradient Search in U-Space
# ============================================================================

def gradient_ascent_u(
    predictor: RankingScorePredictor,
    flow_model: NormalizingFlow,
    init_u: torch.Tensor,
    steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    device: str = 'cuda',
    verbose: bool = False
) -> torch.Tensor:
    """
    Gradient ascent in u-space (prior space) to maximize predicted ranking score.

    Maximizes: R(u) - λ∥u - u₀∥² where R is the u-space trained ranking predictor

    Args:
        predictor: Trained ranking score predictor R(u) (takes u-vectors as input)
        flow_model: Trained normalizing flow F (used for u → z mapping after optimization)
        init_u: Initial u vectors [num_starts, dim]
        steps: Number of gradient ascent steps
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized u vectors [num_starts, dim]
    """
    predictor.eval()
    flow_model.eval()

    u = init_u.clone().to(device).requires_grad_(True)
    u_init = init_u.clone().to(device)  # Keep initial u for trust region
    num_samples = u.shape[0]

    trust_msg = f", trust_region_lambda={trust_region_lambda}" if trust_region_lambda > 0 else ""
    if verbose:
        print(f"Gradient ascent in u-space: {num_samples} starts, {steps} steps, lr={lr}{trust_msg}")

    for step in range(steps):
        if u.grad is not None:
            u.grad.zero_()

        # Predict score R(u) directly - predictor was trained on u-space
        scores = predictor(u)

        # Maximize score (negative for gradient ascent)
        loss = -scores.mean()

        # Add trust region penalty if enabled
        if trust_region_lambda > 0:
            trust_region_penalty = trust_region_lambda * torch.mean((u - u_init) ** 2)
            loss = loss + trust_region_penalty

        loss.backward()

        # Gradient ascent update
        with torch.no_grad():
            u += lr * u.grad

        if verbose and (step + 1) % 20 == 0:
            avg_score = -loss.item()
            if trust_region_lambda > 0:
                with torch.no_grad():
                    distance = torch.mean(torch.sqrt(torch.sum((u - u_init) ** 2, dim=1))).item()
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f} | Dist from init: {distance:.4f}")
            else:
                print(f"  Step {step+1}/{steps} | Avg score: {avg_score:.4f}")

    return u.detach()


def multi_start_gradient_search_u(
    predictor: RankingScorePredictor,
    flow_model: NormalizingFlow,
    num_starts: int = 10,
    steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    init_from_data: torch.Tensor = None,
    device: str = 'cuda',
    verbose: bool = True
) -> torch.Tensor:
    """
    Multi-start gradient search in u-space initialized from existing data.

    Args:
        predictor: Trained ranking predictor
        flow_model: Trained normalizing flow
        num_starts: Number of parallel searches
        steps: Gradient steps per search
        lr: Learning rate
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        init_from_data: Tensor of existing z vectors to initialize from (required)
        device: Device to use
        verbose: Print progress

    Returns:
        Optimized z vectors [num_starts, dim] (mapped back from u-space)
    """
    if init_from_data is None or len(init_from_data) == 0:
        raise ValueError("init_from_data is required - must provide existing z vectors")

    # Initialize from top data points
    n_available = len(init_from_data)
    n_to_use = min(num_starts, n_available)

    # Take top-n embeddings
    init_z = init_from_data[:n_to_use].clone().to(device)

    # Map z → u using flow forward
    with torch.no_grad():
        init_u, _ = flow_model(init_z)

    if verbose:
        print(f"Initialized {n_to_use} searches from top-{n_to_use} embeddings")
        print(f"  Mapped z → u (prior space)")

    # Run gradient ascent in u-space
    optimized_u = gradient_ascent_u(
        predictor=predictor,
        flow_model=flow_model,
        init_u=init_u,
        steps=steps,
        lr=lr,
        trust_region_lambda=trust_region_lambda,
        device=device,
        verbose=verbose
    )

    # Get final scores and sort (predictor takes u as input)
    with torch.no_grad():
        final_scores = predictor(optimized_u).squeeze(-1)  # Only squeeze last dim to keep [num_starts]

    # Map optimized u → z using flow inverse (for decoding)
    with torch.no_grad():
        optimized_z = flow_model.inverse(optimized_u)

    if verbose:
        print(f"\nFinal scores:")
        if final_scores.dim() == 0:
            # Handle single score case
            print(f"  Score: {final_scores.item():.4f}")
        else:
            print(f"  Min:  {final_scores.min().item():.4f}")
            print(f"  Max:  {final_scores.max().item():.4f}")
            print(f"  Mean: {final_scores.mean().item():.4f}")

    # Sort by score (descending)
    # Ensure sorted_indices is 1D to preserve batch dimension when indexing
    if final_scores.dim() == 0:
        # Single score: no sorting needed, keep as 1D index
        sorted_indices = torch.tensor([0], device=final_scores.device)
    else:
        sorted_indices = torch.argsort(final_scores, descending=True)

    optimized_z = optimized_z[sorted_indices]

    return optimized_z


# ============================================================================
# Code Generation (Same as z-space version)
# ============================================================================

def generate_code_from_z(
    z_vector: np.ndarray,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    skeleton_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024
) -> tuple:
    """Generate code from a latent vector z using mapper and decoder."""

    embed_layer = decoder_model.get_input_embeddings()
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    # Ensure z is the right shape
    if isinstance(z_vector, torch.Tensor):
        z_tensor = z_vector.clone().detach().float()
    else:
        z_tensor = torch.tensor(z_vector, dtype=torch.float32)

    if z_tensor.dim() == 1:
        z_tensor = z_tensor.unsqueeze(0)

    z_tensor = z_tensor.to(device)

    with torch.no_grad():
        # Map z to soft prompts
        soft_prompt_embeds = mapper_model(z_tensor).to(device, dtype=dtype)

        # Prepare instruction
        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        instruction_embeds = embed_layer(instruction_ids)

        # Concatenate soft prompts + instruction
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        # Generate
        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )

        raw_output = decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        clean_code = extract_python_code_robust(raw_output, include_preface=True)

        return clean_code, raw_output


def generate_code_from_z_batch(
    z_vectors: torch.Tensor,
    mapper_model,
    decoder_model,
    decoder_tokenizer,
    skeleton_prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 1024
) -> list:
    """
    Generate code from multiple latent vectors in a single batch.

    Args:
        z_vectors: Tensor of shape [batch_size, dim]
        mapper_model: Trained mapper
        decoder_model: Decoder LLM
        decoder_tokenizer: Tokenizer
        skeleton_prompt: Task instruction prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of (clean_code, raw_output) tuples
    """
    embed_layer = decoder_model.get_input_embeddings()
    device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    # Ensure correct shape
    if isinstance(z_vectors, np.ndarray):
        z_vectors = torch.from_numpy(z_vectors)

    if z_vectors.dim() == 1:
        z_vectors = z_vectors.unsqueeze(0)

    batch_size = z_vectors.shape[0]
    z_vectors = z_vectors.to(device).float()

    with torch.no_grad():
        # Batch map z -> soft prompts: [batch_size, num_tokens, hidden_dim]
        soft_prompt_embeds = mapper_model(z_vectors).to(device, dtype=dtype)

        # Prepare instruction (same for all samples)
        instruction_messages = [{"role": "user", "content": skeleton_prompt}]
        instruction_ids = decoder_tokenizer.apply_chat_template(
            instruction_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        # Get instruction embeddings and repeat for batch
        instruction_embeds = embed_layer(instruction_ids)  # [1, seq_len, hidden]
        instruction_embeds = instruction_embeds.expand(batch_size, -1, -1)  # [batch, seq_len, hidden]

        # Concatenate: [batch, num_soft_tokens + seq_len, hidden]
        inputs_embeds = torch.cat([soft_prompt_embeds, instruction_embeds], dim=1)

        # Create attention mask (all 1s since no padding in inputs)
        attention_mask = torch.ones(
            batch_size, inputs_embeds.shape[1],
            dtype=torch.long, device=device
        )

        # Batched generation
        generated_ids = decoder_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=decoder_tokenizer.pad_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id
        )

        # Decode all outputs
        results = []
        for i in range(batch_size):
            raw_output = decoder_tokenizer.decode(generated_ids[i], skip_special_tokens=True)
            clean_code = extract_python_code_robust(raw_output, include_preface=True)
            results.append((clean_code, raw_output))

        return results


# ============================================================================
# LLM Initialization
# ============================================================================

def generate_initial_programs_with_llm(
    task_name: str,
    num_programs: int,
    decoder_model,
    decoder_tokenizer,
    encoder_model,
    evaluator,
    device: str = "cuda",
    temperature: float = 0.8,
    batch_size: int = 4,
    verbose: bool = True
) -> dict:
    """
    Generate initial search points using LLM (decoder) with BATCHED generation.

    Similar to EOH's initialization phase - generates diverse initial programs
    using the decoder model. Uses batched generation for 3-4x speedup.

    Args:
        task_name: Task to generate programs for
        num_programs: Number of initial programs to generate
        decoder_model: Decoder LLM for generation
        decoder_tokenizer: Tokenizer for decoder
        encoder_model: Encoder for embedding programs
        evaluator: SecureEvaluator for scoring
        device: Device to use
        temperature: Sampling temperature (higher = more diverse)
        batch_size: Number of programs to generate in parallel (default: 4)
        verbose: Print progress

    Returns:
        Dictionary: {code_hash: {'code', 'actual_score', 'embedding', 'source', 'iteration'}}
    """
    print(f"\n{'='*70}")
    print(f"LLM Initialization: Generating {num_programs} Initial Programs (BATCHED)")
    print(f"{'='*70}\n")

    # Load task template
    template_program, task_description = load_task_template(task_name)
    if template_program is None or task_description is None:
        raise ValueError(f"Could not load template for task: {task_name}")

    # Create initialization prompt
    init_prompt = create_init_prompt(task_description, template_program)
    if verbose:
        print(f"Initialization prompt:\n{init_prompt[:500]}...\n")

    embed_layer = decoder_model.get_input_embeddings()
    decoder_device = embed_layer.weight.device
    dtype = embed_layer.weight.dtype

    program_database = {}
    generated_count = 0
    valid_count = 0

    # Generate more than needed since some will fail validation
    total_attempts = num_programs * 3

    print(f"Generating programs in batches (target: {num_programs}, batch_size: {batch_size})...")

    # Prepare prompt once (same for all samples in LLM init)
    messages = [{"role": "user", "content": init_prompt}]
    prompt_text = decoder_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Generate in batches
    num_batches = (total_attempts + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        if valid_count >= num_programs:
            break

        # Determine actual batch size for this iteration
        current_batch_size = min(batch_size, total_attempts - batch_idx * batch_size)

        try:
            # Tokenize batch (same prompt repeated)
            # Note: We repeat the same prompt, but do_sample=True with temperature
            # ensures different outputs due to stochastic sampling
            batch_prompts = [prompt_text] * current_batch_size

            tokenized = decoder_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False
            ).to(decoder_device)

            # Batched generation
            with torch.no_grad():
                generated_ids = decoder_model.generate(
                    **tokenized,
                    max_new_tokens=1024,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,  # Ensures diversity despite same prompt
                    pad_token_id=decoder_tokenizer.pad_token_id,
                    eos_token_id=decoder_tokenizer.eos_token_id
                )

            # Process each generated output in the batch
            for i in range(current_batch_size):
                generated_count += 1

                # Decode
                raw_output = decoder_tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                clean_code = extract_python_code_robust(raw_output, include_preface=True)

                # Validate syntax
                if not clean_code or not is_valid_python(clean_code):
                    continue

                # Parse function
                program = TextFunctionProgramConverter.text_to_program(clean_code)
                if program is None or len(program.functions) == 0:
                    continue

                # Check for duplicates
                code_hash = hash(clean_code.strip())
                if code_hash in program_database:
                    continue

                # Evaluate
                try:
                    score = evaluator.evaluate_program(clean_code)
                    if score is None or not np.isfinite(score):
                        continue
                except Exception as e:
                    if verbose and batch_idx == 0 and i == 0:  # Only print once
                        print(f"  Evaluation errors may occur (first: {e})")
                    continue

                # Encode
                with torch.no_grad():
                    z_embedding = encoder_model.encode(
                        [clean_code],
                        convert_to_tensor=True,
                        device=device,
                        show_progress_bar=False
                    )[0].float()

                # Add to database
                program_database[code_hash] = {
                    'code': clean_code,
                    'actual_score': float(score),
                    'embedding': z_embedding,
                    'source': 'llm_init',
                    'iteration': -1
                }

                valid_count += 1

                if verbose and valid_count % 5 == 0:
                    print(f"  Generated {valid_count}/{num_programs} valid programs")

                # Early exit if we have enough
                if valid_count >= num_programs:
                    break

        except Exception as e:
            if verbose:
                print(f"  Batch generation error: {e}")
            continue

    print(f"\n✓ LLM Initialization Complete:")
    print(f"  Total attempts: {generated_count}")
    print(f"  Valid programs: {valid_count}")
    print(f"  Speedup: ~{batch_size}x faster with batching")

    if valid_count > 0:
        scores = [p['actual_score'] for p in program_database.values()]
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Worst score: {min(scores):.4f}")
        print(f"  Mean score: {np.mean(scores):.4f}")

    return program_database


# ============================================================================
# Main Search Pipeline
# ============================================================================

def gradient_search_pipeline_u(
    task_name: str,
    predictor_path: str,
    flow_path: str,
    mapper_path: str,
    num_iterations: int = 5,
    num_searches_per_iter: int = 10,
    top_k_pool_size: int = 30,
    gradient_steps: int = 100,
    lr: float = 0.01,
    trust_region_lambda: float = 0.0,
    temperature: float = 0.7,
    encoder_name: str = None,
    decoder_name: str = None,
    embedding_dim: int = None,
    device: str = "cuda",
    output_dir: str = "gradient_search_u_results",
    num_evaluators: int = 4,
    generation_batch_size: int = 4,
    llm_init: bool = False,
    llm_init_count: int = 20,
    verbose: bool = True
):
    """
    Full gradient search pipeline in u-space (prior space).

    Key Features:
    1. Evaluates ALL baseline heuristics on ACTUAL test instances (not predicted scores)
    2. Maintains database of all programs (baselines + discoveries) with actual scores
    3. Each iteration: sorts database by actual score, samples from top-k pool using softmax probabilities
    4. Guarantees starting from PROVEN best programs, not predicted best
    5. Softmax-weighted sampling: better programs have higher selection probability, but exploration is maintained
    6. No repetition in sampling (samples without replacement)
    7. Batched code generation for efficiency
    8. Parallel evaluation using ThreadPoolExecutor

    Args:
        task_name: Task to search for (e.g., 'tsp_construct')
        predictor_path: Path to trained ranking predictor
        flow_path: Path to trained normalizing flow
        mapper_path: Path to trained mapper
        num_iterations: Number of search iterations
        num_searches_per_iter: Number of gradient searches per iteration (how many to sample)
        top_k_pool_size: Pool size to sample seeds from (e.g., 30 means sample from best 30 programs)
        gradient_steps: Steps per gradient search
        lr: Learning rate for gradient ascent
        trust_region_lambda: Trust region penalty coefficient (0 = disabled)
        temperature: Sampling temperature for decoder
        encoder_name: Encoder model name. Defaults to DEFAULT_ENCODER from model_config.py
        decoder_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
        device: Device to use
        output_dir: Directory to save results
        num_evaluators: Number of parallel evaluation workers (default: 4)
        generation_batch_size: Batch size for code generation (default: 4, adjust based on GPU memory)
        llm_init: If True, use LLM to generate initial programs instead of loading from JSON
        llm_init_count: Number of programs to generate for LLM initialization (default: 20)
        verbose: Print progress
    """

    # Set defaults from config
    if encoder_name is None:
        encoder_name = DEFAULT_ENCODER
    if decoder_name is None:
        decoder_name = DEFAULT_DECODER

    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("Gradient Search in U-Space (Prior Space)")
    print("="*70)
    print(f"Task: {task_name}")
    print(f"Predictor: {predictor_path}")
    print(f"Flow: {flow_path}")
    print(f"Mapper: {mapper_path}")
    print(f"Encoder: {encoder_name}")
    print(f"Decoder: {decoder_name}")
    print(f"Parallel evaluators: {num_evaluators}")
    print(f"Generation batch size: {generation_batch_size}")
    print(f"LLM initialization: {llm_init}" + (f" ({llm_init_count} programs)" if llm_init else ""))
    print()

    # ===== Load Models =====

    # Load ranking predictor
    print("Loading ranking predictor...")
    predictor, predictor_info = load_ranking_predictor(predictor_path, device)
    predictor.to(device)
    predictor.eval()

    # Load normalizing flow
    flow_model = load_flow(flow_path, device)

    # Load decoder
    decoder_model, decoder_tokenizer = load_decoder(decoder_name, device)

    # Load mapper
    mapper_model = load_mapper(mapper_path, decoder_model, device)
    embed_layer = decoder_model.get_input_embeddings()
    mapper_model = mapper_model.to(embed_layer.weight.device)

    # Load encoder ONCE
    print(f"Loading encoder ({encoder_name})...")
    encoder_model, actual_embedding_dim = get_encoder_model(device, encoder_name, embedding_dim)
    print(f"Embedding dimension: {actual_embedding_dim}")

    # Load evaluator
    print(f"Loading evaluator for {task_name}...")
    evaluator = get_evaluator(task_name)
    secure_eval = SecureEvaluator(evaluator, debug_mode=False)

    # Get skeleton prompt
    skeleton_prompt = TASK_PROMPTS.get(task_name)
    if skeleton_prompt is None:
        raise ValueError(f"No prompt defined for task: {task_name}")

    # ===== Load and Evaluate Initial Programs =====

    # Initialize program database: {code_hash: {'code', 'actual_score', 'embedding', 'source'}}
    program_database = {}

    if llm_init:
        # Use LLM to generate initial programs (BATCHED for speed)
        program_database = generate_initial_programs_with_llm(
            task_name=task_name,
            num_programs=llm_init_count,
            decoder_model=decoder_model,
            decoder_tokenizer=decoder_tokenizer,
            encoder_model=encoder_model,
            evaluator=secure_eval,
            device=device,
            temperature=0.8,  # Higher temp for diversity in initialization
            batch_size=generation_batch_size,  # Use same batch size as gradient search
            verbose=verbose
        )

        if len(program_database) == 0:
            raise ValueError("LLM initialization failed - no valid programs generated!")

    else:
        # Load from JSON file (original behavior)
        print(f"\nLoading baseline heuristics from task/{task_name}/heuristics.json...")
        heuristics = load_heuristics(task_name)
        baseline_codes = list(heuristics.values())

        if len(baseline_codes) == 0:
            raise ValueError(f"No heuristics found in task/{task_name}/heuristics.json")

        print(f"  Loaded {len(baseline_codes)} baseline heuristics")

        # Evaluate ALL baselines to get ACTUAL scores (not predicted!) - PARALLEL
        print(f"\nEvaluating baseline heuristics on actual test instances (parallel, {num_evaluators} workers)...")
        print("(This may take a few minutes but only runs once)")

        def eval_baseline(idx_code):
            """Evaluate a single baseline heuristic."""
            idx, code = idx_code
            try:
                score = secure_eval.evaluate_program(code)
                return idx, code, score, None
            except Exception as e:
                return idx, code, None, str(e)

        # Submit all baseline evaluations in parallel
        baseline_results = []
        with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
            futures = {executor.submit(eval_baseline, (idx, code)): idx
                       for idx, code in enumerate(baseline_codes)}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating baselines"):
                baseline_results.append(future.result())

        # Process results and encode (encoding stays sequential - GPU bound)
        valid_baselines = [(idx, code, score) for idx, code, score, error in baseline_results
                           if score is not None and np.isfinite(score)]

        print(f"  Valid baselines: {len(valid_baselines)}/{len(baseline_codes)}")
        print("  Encoding valid baselines...")

        for idx, code, score in tqdm(valid_baselines, desc="Encoding baselines"):
            try:
                # Encode
                with torch.no_grad():
                    z_embedding = encoder_model.encode(
                        [code],
                        convert_to_tensor=True,
                        device=device,
                        show_progress_bar=False
                    )[0].float()

                # Add to database
                code_hash = hash(code.strip())
                program_database[code_hash] = {
                    'code': code,
                    'actual_score': float(score),
                    'embedding': z_embedding,
                    'source': 'baseline',
                    'iteration': -1
                }

            except Exception as e:
                if verbose:
                    print(f"  Baseline {idx+1}: Encoding error - {e}")
                continue

        if len(program_database) == 0:
            raise ValueError("No valid baseline programs after evaluation!")

    # Show statistics
    sorted_programs = sorted(program_database.values(), key=lambda x: x['actual_score'], reverse=True)
    source_type = "LLM-generated" if llm_init else "baseline"
    print(f"\n✓ Successfully initialized {len(program_database)} {source_type} programs")
    print(f"  Best {source_type} score: {sorted_programs[0]['actual_score']:.4f}")
    print(f"  Worst {source_type} score: {sorted_programs[-1]['actual_score']:.4f}")
    print(f"  Mean {source_type} score: {np.mean([p['actual_score'] for p in program_database.values()]):.4f}")

    # ===== Run Search =====

    all_results = []
    successful_programs = {}

    for iteration in range(num_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*70}")

        # Sort ALL programs (baselines + discoveries) by ACTUAL score
        # Higher score is better (evaluators return negative costs)
        sorted_programs = sorted(
            program_database.values(),
            key=lambda x: x['actual_score'],
            reverse=True  # DESCENDING - higher (less negative) is better
        )

        # Create top-k pool and sample seeds based on softmax probabilities
        # This adds stochasticity to avoid getting stuck in same neighborhoods
        # while favoring better programs (higher scores have higher selection probability)
        pool_size = min(top_k_pool_size, len(sorted_programs))
        top_k_pool = sorted_programs[:pool_size]

        num_seeds = min(num_searches_per_iter, pool_size)

        # Compute softmax probabilities based on scores
        # Higher score (less negative) = higher probability
        pool_scores = np.array([p['actual_score'] for p in top_k_pool])

        # Apply softmax: exp(score) / sum(exp(score))
        # Use temperature=1.0 for standard softmax (can be tuned for more/less exploration)
        exp_scores = np.exp(pool_scores - np.max(pool_scores))  # Subtract max for numerical stability
        probabilities = exp_scores / np.sum(exp_scores)

        # Sample from top-k pool using softmax probabilities (no replacement)
        sampled_indices = np.random.choice(pool_size, num_seeds, replace=False, p=probabilities)
        top_k_programs = [top_k_pool[i] for i in sampled_indices]

        # Extract embeddings from sampled programs
        init_embeddings = torch.stack([p['embedding'] for p in top_k_programs])

        # Get score range and probabilities for logging
        sampled_scores = [top_k_programs[i]['actual_score'] for i in range(num_seeds)]
        sampled_probs = [probabilities[i] for i in sampled_indices]
        best_sampled = max(sampled_scores)
        worst_sampled = min(sampled_scores)

        print(f"\nStarting gradient search by sampling {num_seeds} from top-{pool_size} programs (softmax-weighted):")
        print(f"  Best score in database: {sorted_programs[0]['actual_score']:.4f} ({sorted_programs[0]['source']})")
        print(f"  Best in sampled seeds: {best_sampled:.4f} (P={max(sampled_probs):.3f})")
        print(f"  Worst in sampled seeds: {worst_sampled:.4f} (P={min(sampled_probs):.3f})")
        print(f"  Database size: {len(program_database)} programs")

        # Gradient search in u-space from top-k seeds
        print(f"\nRunning gradient search in u-space ({num_seeds} starts, {gradient_steps} steps)...")

        optimized_z = multi_start_gradient_search_u(
            predictor=predictor,
            flow_model=flow_model,
            num_starts=num_seeds,
            steps=gradient_steps,
            lr=lr,
            trust_region_lambda=trust_region_lambda,
            init_from_data=init_embeddings,
            device=device,
            verbose=verbose
        )

        # ===== BATCHED GENERATION =====
        print(f"\nGenerating {len(optimized_z)} candidates in batches (batch_size={generation_batch_size})...")

        all_generated = []
        for batch_start in range(0, len(optimized_z), generation_batch_size):
            batch_end = min(batch_start + generation_batch_size, len(optimized_z))
            z_batch = optimized_z[batch_start:batch_end]

            batch_results = generate_code_from_z_batch(
                z_batch,
                mapper_model,
                decoder_model,
                decoder_tokenizer,
                skeleton_prompt,
                temperature=temperature
            )
            all_generated.extend(batch_results)

            if verbose:
                print(f"  Generated batch {batch_start//generation_batch_size + 1}/{(len(optimized_z) + generation_batch_size - 1)//generation_batch_size}")

        # ===== VALIDATE AND FILTER =====
        print(f"Validating {len(all_generated)} generated programs...")

        valid_candidates = []
        existing_hashes = set(hash(c.strip()) for c in successful_programs.values())

        for idx, (clean_code, raw_output) in enumerate(all_generated):
            # Validate syntax
            if not clean_code or not is_valid_python(clean_code):
                continue

            # Parse function
            program = TextFunctionProgramConverter.text_to_program(clean_code)
            if program is None or len(program.functions) == 0:
                continue

            func_name = program.functions[0].name

            # Check for duplicates
            code_hash = hash(clean_code.strip())
            if code_hash in existing_hashes:
                continue

            existing_hashes.add(code_hash)
            valid_candidates.append((idx, clean_code, func_name))

        print(f"  Valid candidates: {len(valid_candidates)}/{len(all_generated)}")

        # ===== PARALLEL EVALUATION =====
        print(f"Evaluating {len(valid_candidates)} candidates in parallel ({num_evaluators} workers)...")

        def eval_candidate(item):
            """Evaluate a single candidate."""
            idx, clean_code, func_name = item
            try:
                score = secure_eval.evaluate_program(clean_code)
                return idx, clean_code, func_name, score, None
            except Exception as e:
                return idx, clean_code, func_name, None, str(e)

        eval_results = []
        with ThreadPoolExecutor(max_workers=num_evaluators) as executor:
            futures = {executor.submit(eval_candidate, item): item[0]
                       for item in valid_candidates}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating candidates"):
                eval_results.append(future.result())

        # ===== PROCESS RESULTS =====
        successful = 0
        failed = len(all_generated) - len(valid_candidates)  # Already failed validation

        for idx, clean_code, func_name, score, error in eval_results:
            if score is None or not np.isfinite(score):
                failed += 1
                continue

            # Success! Add to database with ACTUAL score
            program_id = f"iter{iteration}_idx{idx}"
            successful_programs[program_id] = clean_code

            # Add to program database
            code_hash_new = hash(clean_code.strip())
            if code_hash_new not in program_database:
                # Encode the new program
                with torch.no_grad():
                    z_new = encoder_model.encode(
                        [clean_code],
                        convert_to_tensor=True,
                        device=device,
                        show_progress_bar=False
                    )[0].float()

                program_database[code_hash_new] = {
                    'code': clean_code,
                    'actual_score': float(score),
                    'embedding': z_new,
                    'source': 'discovered',
                    'iteration': iteration
                }

            all_results.append({
                'program_id': program_id,
                'iteration': iteration,
                'score': score,
                'code': clean_code,
                'function_name': func_name
            })

            successful += 1

            if verbose:
                print(f"  ✓ Score: {score:.4f} | Function: {func_name}")

        # Iteration summary
        print(f"\nIteration {iteration + 1} Summary:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total programs in database: {len(program_database)}")
        print(f"  Total discovered (all iterations): {len(successful_programs)}")

        if all_results:
            scores = [r['score'] for r in all_results]
            print(f"  Best score so far: {max(scores):.4f}")  # max - higher is better!
            print(f"  Avg score (discoveries): {np.mean(scores):.4f}")

        # Database statistics
        db_scores = [p['actual_score'] for p in program_database.values()]
        print(f"  Best score in database: {max(db_scores):.4f}")  # max - higher is better!
        print(f"  Database programs: {sum(1 for p in program_database.values() if p['source']=='baseline')} baselines, "
              f"{sum(1 for p in program_database.values() if p['source']=='discovered')} discovered")

    # ===== Final Results =====

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal programs in database: {len(program_database)}")
    print(f"  Baselines: {sum(1 for p in program_database.values() if p['source']=='baseline')}")
    print(f"  Discovered: {sum(1 for p in program_database.values() if p['source']=='discovered')}")

    # Get all database scores
    db_programs = list(program_database.values())
    db_programs.sort(key=lambda x: x['actual_score'], reverse=True)  # DESCENDING - higher is better

    print(f"\nDatabase Statistics (ACTUAL scores):")
    print(f"  Best score: {db_programs[0]['actual_score']:.4f} ({db_programs[0]['source']}, iter {db_programs[0]['iteration']})")
    print(f"  Worst score: {db_programs[-1]['actual_score']:.4f}")
    print(f"  Mean score: {np.mean([p['actual_score'] for p in db_programs]):.4f}")
    print(f"  Std score: {np.std([p['actual_score'] for p in db_programs]):.4f}")

    if all_results:
        # Sort discovered programs by score (DESCENDING - higher is better)
        all_results.sort(key=lambda x: x['score'], reverse=True)

        scores = [r['score'] for r in all_results]
        print(f"\nDiscovered Programs Only:")
        print(f"  Count: {len(all_results)}")
        print(f"  Best score: {max(scores):.4f}")
        print(f"  Avg score: {np.mean(scores):.4f}")
        print(f"  Std score: {np.std(scores):.4f}")

        # Show top 5 discovered programs
        print(f"\nTop 5 Discovered Programs:")
        print("-"*70)
        for i, result in enumerate(all_results[:5]):
            print(f"\n{i+1}. Score: {result['score']:.4f} | {result['function_name']}")
            print(f"   ID: {result['program_id']}")
            code_preview = result['code'][:200].replace('\n', '\n   ')
            print(f"   {code_preview}...")

    # Show top 5 from entire database (baselines + discoveries)
    print(f"\nTop 5 Programs Overall (Baselines + Discoveries):")
    print("-"*70)
    for i, prog in enumerate(db_programs[:5]):
        print(f"\n{i+1}. Score: {prog['actual_score']:.4f} | Source: {prog['source']} | Iter: {prog['iteration']}")
        code_preview = prog['code'][:200].replace('\n', '\n   ')
        print(f"   {code_preview}...")

    # ===== Save Results =====

    # Save discovered programs only
    programs_path = os.path.join(output_dir, f"{task_name}_gradient_searched_u.json")
    with open(programs_path, 'w') as f:
        json.dump(successful_programs, f, indent=2)
    print(f"\nSaved {len(successful_programs)} discovered programs to {programs_path}")

    # Save detailed results for discoveries
    results_path = os.path.join(output_dir, f"{task_name}_search_results_u.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved detailed discovery results to {results_path}")

    # Save entire database (baselines + discoveries) with actual scores
    database_export = {}
    for code_hash, prog_data in program_database.items():
        # Create unique name
        if prog_data['source'] == 'baseline':
            name = f"baseline_{code_hash}"
        else:
            name = f"discovered_iter{prog_data['iteration']}_{code_hash}"

        database_export[name] = {
            'code': prog_data['code'],
            'actual_score': prog_data['actual_score'],
            'source': prog_data['source'],
            'iteration': prog_data['iteration']
        }

    database_path = os.path.join(output_dir, f"{task_name}_full_database_u.json")
    with open(database_path, 'w') as f:
        json.dump(database_export, f, indent=2)
    print(f"Saved full database ({len(database_export)} programs) to {database_path}")

    # Clean up
    del encoder_model
    torch.cuda.empty_cache()

    return all_results, successful_programs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Gradient search in u-space (prior space)')
    parser.add_argument('--task', type=str, default='tsp_construct', help='Task name')
    parser.add_argument('--predictor', type=str, default='ranking_predictor_u.pth', help='Path to u-space ranking predictor')
    parser.add_argument('--flow', type=str, default='Flow_Checkpoints/unified_flow_final.pth', help='Path to normalizing flow')
    parser.add_argument('--mapper', type=str, default='Mapper_Checkpoints/unified_mapper.pth', help='Path to mapper')
    parser.add_argument('--encoder', type=str, default=DEFAULT_ENCODER, help=f'Encoder model (default: {DEFAULT_ENCODER})')
    parser.add_argument('--embedding-dim', type=int, default=None,
                        help=f'Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or "model native"})')
    parser.add_argument('--decoder', type=str, default=DEFAULT_DECODER, help=f'Decoder model (default: {DEFAULT_DECODER})')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of search iterations')
    parser.add_argument('--num_searches', type=int, default=10, help='Searches per iteration (how many seeds to sample)')
    parser.add_argument('--top_k_pool', type=int, default=30, help='Pool size for softmax-weighted sampling (e.g., 30 = sample from top 30 programs using softmax probabilities)')
    parser.add_argument('--gradient_steps', type=int, default=100, help='Gradient steps per search')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--trust_region_lambda', type=float, default=0.0, help='Trust region penalty coefficient (0 = disabled)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Decoder temperature')
    parser.add_argument('--output_dir', type=str, default='gradient_search_u_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_evaluators', type=int, default=4, help='Number of parallel evaluation workers')
    parser.add_argument('--generation_batch_size', type=int, default=4, help='Batch size for code generation (adjust based on GPU memory)')
    parser.add_argument('--llm_init', action='store_true', default=False,
                        help='Use LLM to generate initial programs instead of loading from JSON')
    parser.add_argument('--llm_init_count', type=int, default=20,
                        help='Number of programs to generate for LLM initialization (default: 20)')

    args = parser.parse_args()

    results, programs = gradient_search_pipeline_u(
        task_name=args.task,
        predictor_path=args.predictor,
        flow_path=args.flow,
        mapper_path=args.mapper,
        num_iterations=args.num_iterations,
        num_searches_per_iter=args.num_searches,
        top_k_pool_size=args.top_k_pool,
        gradient_steps=args.gradient_steps,
        lr=args.lr,
        trust_region_lambda=args.trust_region_lambda,
        temperature=args.temperature,
        encoder_name=args.encoder,
        decoder_name=args.decoder,
        embedding_dim=getattr(args, 'embedding_dim', None),
        device=args.device,
        output_dir=args.output_dir,
        num_evaluators=args.num_evaluators,
        generation_batch_size=args.generation_batch_size,
        llm_init=args.llm_init,
        llm_init_count=args.llm_init_count,
        verbose=True
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
