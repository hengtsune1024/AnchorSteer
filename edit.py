"""
Structure-preserving music editing using AnchorSteer + MuseControlLite.

Extracts melody, rhythm, and dynamics from a source WAV, then generates a new
audio clip that preserves those structural properties while applying the concept
steering defined by --concept_dir.
"""
import argparse
import os
from MuseControlLite.inference_utils import Inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply concept steering to a source audio file while preserving musical structure."
    )
    parser.add_argument(
        "--source_audio", type=str, required=True,
        help="Path to the source WAV file whose structure should be preserved."
    )
    parser.add_argument(
        "--concept_dir", type=str, required=True,
        help="Path to a trained AnchorSteer checkpoint directory (must contain best.pth and config.yaml)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory where the edited WAV will be saved."
    )
    parser.add_argument(
        "--prompt", type=str, default="a music piece",
        help="Text prompt describing the desired output."
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="Negative text prompt."
    )
    parser.add_argument(
        "--num_steps", type=int, default=50,
        help="Number of diffusion sampling steps."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use_last", action="store_true",
        help="Load the final-epoch checkpoint (adaptor.pth) instead of best.pth."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.source_audio):
        raise FileNotFoundError(f"Source audio not found: {args.source_audio}")
    if not os.path.isdir(args.concept_dir):
        raise FileNotFoundError(
            f"Concept directory not found: {args.concept_dir}\n"
            "Download pre-trained checkpoints or train a new Controller with train.py first."
        )

    infer = Inference(use_best=not args.use_last)
    out_path = infer.infer_single(
        wav_path=args.source_audio,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_dir=args.output_dir,
        concept_dir=args.concept_dir,
        seed=args.seed,
        num_steps=args.num_steps,
    )
    print(f"Saved edited audio to: {out_path}")


if __name__ == "__main__":
    main()
