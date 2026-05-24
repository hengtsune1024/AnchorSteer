import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight Controller to steer Stable Audio Open generation toward a target concept."
    )

    # ── Reproducibility & environment ────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=1234,
        help="Global random seed for reproducible training.",
    )
    parser.add_argument(
        "--logging_dir", type=str, default="logs",
        help="Subdirectory inside output_dir where Accelerate writes experiment logs.",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
        help="Mixed-precision mode. 'bf16' is recommended for RTX 30xx/40xx; 'fp16' for older GPUs.",
    )

    # ── Training schedule ─────────────────────────────────────────────────────
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help=(
            "Hard cap on the total number of optimizer steps. "
            "If omitted, training runs for num_train_epochs full epochs."
        ),
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=20,
        help="Number of full passes over the training dataset.",
    )
    parser.add_argument(
        "--mini_batch_size", type=int, default=2,
        help=(
            "Number of pre-encoded latent samples per device per gradient step. "
            "Effective batch size = mini_batch_size × gradient_accumulation_steps."
        ),
    )
    parser.add_argument(
        "--encode_batch_size", type=int, default=4,
        help=(
            "Batch size used when VAE-encoding the raw audio dataset during preprocessing. "
            "Does not affect training dynamics; increase for faster preprocessing on high-VRAM GPUs."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2,
        help=(
            "Accumulate gradients over this many forward passes before one optimizer update. "
            "Effective batch size = mini_batch_size × gradient_accumulation_steps."
        ),
    )

    # ── Data & output paths ───────────────────────────────────────────────────
    parser.add_argument(
        "--train_data_dir", type=str, default="datasets/person",
        help="Directory containing pre-generated reference audio files (*.wav) and labels.json.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="exps/exp_person",
        help="Directory where checkpoints (best.pth, adaptor.pth), config.yaml, and loss plots are saved.",
    )

    # ── Controller architecture ───────────────────────────────────────────────
    parser.add_argument(
        "--control_type", type=str, default="transformer",
        help=(
            "Controller architecture. "
            "'vector': a single learnable delta matrix applied uniformly to all time positions (~1–3 M params, fast). "
            "'transformer': bottleneck TransformerEncoder that captures temporal context across time positions (more expressive, slower)."
        ),
    )
    parser.add_argument(
        "--train_full_len", action="store_true",
        help=(
            "Train on the full 47-second audio clip. "
            "If omitted, only the first 5 seconds are used (107 latent frames instead of 1024). "
            "This flag also determines the Controller's hidden-state sequence dimension, so it must match at inference."
        ),
    )
    parser.add_argument(
        "--use_time_emb", action="store_true",
        help=(
            "Condition the Controller on the diffusion timestep embedding. "
            "A small MLP maps the timestep embedding to (gamma, beta) and applies an affine modulation to the delta, "
            "allowing the Controller to behave differently at different noise levels."
        ),
    )
    parser.add_argument(
        "--inject_first_state", action="store_true",
        help=(
            "Include position 0 (the prepended audio-duration global token) in the injected region. "
            "By default injection starts at position 1 to leave the duration conditioning untouched."
        ),
    )
    parser.add_argument(
        "--edit_start", type=int, default=0,
        help="Index of the first SAO DiT transformer block (0–23) to inject the Controller delta into.",
    )
    parser.add_argument(
        "--edit_end", type=int, default=23,
        help="Index of the last SAO DiT transformer block (0–23, inclusive) to inject the Controller delta into.",
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4,
        help="Peak learning rate for AdamW (reached after the warmup period).",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine",
        help=(
            "Learning-rate schedule. "
            'Choices: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup".'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100,
        help="Number of steps to linearly ramp the learning rate from 0 to learning_rate.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="AdamW beta1 (exponential decay rate for the first moment estimate).",
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="AdamW beta2 (exponential decay rate for the second moment estimate).",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-4,
        help="AdamW weight-decay (L2 regularization coefficient).",
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="AdamW epsilon for numerical stability in the denominator.",
    )

    # ── Checkpointing & logging ───────────────────────────────────────────────
    parser.add_argument(
        "--skip_evaluation", action="store_true",
        help="Skip the periodic mid-training checkpoint save and loss-plot update (saves time when iterating quickly).",
    )
    parser.add_argument(
        "--log_every_steps", type=int, default=1000,
        help="Save adaptor.pth and refresh loss_history.png every N optimizer steps (only when --skip_evaluation is not set).",
    )
    parser.add_argument(
        "--log_every_epochs", type=int, default=5,
        help="Save adaptor.pth every N epochs regardless of skip_evaluation.",
    )
    parser.add_argument(
        "--num_inference_steps", default=50, type=int,
        help="Number of diffusion sampling steps used when generating validation audio during training.",
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
