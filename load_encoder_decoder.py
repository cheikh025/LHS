
import os
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from model_config import DEFAULT_ENCODER, DEFAULT_DECODER, DEFAULT_MATRYOSHKA_DIM


def load_encoder(model_name: str = None, device: str = "cuda", truncate_dim: int = None):
    """Load encoder with same config as training (fp16, optional Matryoshka truncation).
    Args:
        model_name: Encoder model name. Defaults to DEFAULT_ENCODER from model_config.py
        device: Device to load the model on.
        truncate_dim: Matryoshka dimension to truncate embeddings to.
            If None, uses DEFAULT_MATRYOSHKA_DIM from model_config.py.
            If that is also None, uses the model's native dimension.

    Returns:
        Tuple of (encoder_model, embedding_dim) where embedding_dim is the
        actual dimension of output embeddings.
    """
    if model_name is None:
        model_name = DEFAULT_ENCODER

    # Use provided truncate_dim, fall back to config default
    if truncate_dim is None:
        truncate_dim = DEFAULT_MATRYOSHKA_DIM

    encoder_model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.float16},
        truncate_dim=truncate_dim,
    ).to(device)
    encoder_model.eval()

    # Get actual embedding dimension
    if truncate_dim is not None:
        embedding_dim = truncate_dim
    else:
        # Get native dimension from model
        embedding_dim = encoder_model.get_sentence_embedding_dimension()

    return encoder_model, embedding_dim


def load_decoder(model_name: str = None, device: str = "auto"):
    """Load decoder with Flash Attention 2.

    Args:
        model_name: Decoder model name. Defaults to DEFAULT_DECODER from model_config.py
        device: Device map for model placement. Defaults to "auto".
    """
    if model_name is None:
        model_name = DEFAULT_DECODER

    decoder_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    decoder_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
        decoder_tokenizer.pad_token_id = decoder_tokenizer.eos_token_id

    return decoder_model, decoder_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load encoder + decoder models")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, help=f"Encoder model name (default: {DEFAULT_ENCODER})")
    parser.add_argument("--decoder", type=str, default=DEFAULT_DECODER, help=f"Decoder model name (default: {DEFAULT_DECODER})")
    parser.add_argument("--device", type=str, default="cuda", help="Device for encoder")
    parser.add_argument("--embedding-dim", type=int, default=None,
                        help=f"Matryoshka embedding dimension (default: {DEFAULT_MATRYOSHKA_DIM or 'model native'})")
    args = parser.parse_args()

    # Encoder (uses same config as training)
    if args.encoder != DEFAULT_ENCODER:
        print(f"Warning: encoder name differs from training default ({DEFAULT_ENCODER}).")
    encoder, embedding_dim = load_encoder(
        model_name=args.encoder,
        device=args.device,
        truncate_dim=args.embedding_dim
    )

    # Decoder (uses same config as training)
    decoder, tokenizer = load_decoder(model_name=args.decoder)

    print("Loaded encoder:", args.encoder, "on", args.device)
    print("Embedding dimension:", embedding_dim)
    print("Loaded decoder:", args.decoder)
    print("Decoder dtype:", decoder.dtype)


if __name__ == "__main__":
    main()
