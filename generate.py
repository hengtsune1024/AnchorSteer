import logging
import os
import torch
from ruamel.yaml import YAML
import argparse
import soundfile as sf

from models import Controller, load_pipeline

logger = logging.getLogger(__name__)

def main(args):
    yaml = YAML()
    with open(os.path.join(args.exp_dir, 'config.yaml'), "r", encoding="utf-8") as f:
        configs = yaml.load(f)

    # Merge training config into args, but only for keys not already supplied on the
    # command line — CLI flags always take priority over the stored config.
    for k, v in configs.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.output_dir = args.exp_dir

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    target_dtype = torch.float32
    device = args.device

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info(f"Loading pipeline and controller from {args.exp_dir}")
    logger.info("Loading Stable Audio pipeline...")
    pipe = load_pipeline("interpret", dtype=target_dtype)
    # inject_first_state was added after the initial release; old config.yaml files won't
    # have this key, so getattr with a default is needed for backward compatibility.
    num_hidden_vec = (1024 if args.train_full_len else 107) + (1 if getattr(args, "inject_first_state", False) else 0)
    controller = Controller(
        edit_layers=list(range(args.edit_start, args.edit_end+1)),
        use_time=args.use_time_emb,
        total_layers=len(pipe.transformer.transformer_blocks),
        control_type=args.control_type,
        inject_first_state=getattr(args, "inject_first_state", False),
        hidden_dim=(num_hidden_vec, 1536)
    )
    pipe.set_controller(controller)
    print(f"Controller dimension: {(num_hidden_vec, 1536)}")

    model_path = os.path.join(args.output_dir, 'adaptor.pth' if args.use_last else 'best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Download pre-trained checkpoints from the project page and extract to exps/, "
            "or train a new Controller with train.py first."
        )
    controller.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
    controller.eval()
    logger.info(f"Loading trained weights from {model_path}")

    pipe = pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.projection_model.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    prompt = args.prompt

    save_audio_dir = os.path.join(args.output_dir, "last" if args.use_last else "best")
    os.makedirs(save_audio_dir, exist_ok=True)

    sample_rate = pipe.vae.sampling_rate
    if args.audio_len is not None:
        # audio_len <= 0 is treated as "full length" to match data_creation.py convention
        audio_len = (args.audio_len if args.audio_len > 0 else None)
    else:
        audio_len = (None if args.train_full_len else 5.0)

    logger.info(f"Generating {args.num_sample} samples per prompt...")

    for j in range(args.num_sample):
        # Seeds are offset from 100 to stay clear of the seeds used during dataset
        # generation (data_creation.py starts at 0), keeping generations independent.
        seed = 100 + j
        
        logger.info(f"Generating: '{prompt}' original, seed: {seed}")
        with torch.no_grad():
            output = pipe(
                prompt=[prompt],
                num_inference_steps=args.num_inference_steps,
                num_waveforms_per_prompt=1,
                audio_start_in_s=0.0,
                audio_end_in_s=audio_len,
                generator=torch.Generator(device=device).manual_seed(int(seed)),
                output_type="np",
            )
        orig_path = os.path.join(save_audio_dir, f"{prompt}_{j}_orig.wav")
        sf.write(orig_path, output.audios[0].T, sample_rate)
        logger.info(f"Saved to {orig_path}")

        logger.info(f"Generating: '{prompt}' steered, seed: {seed}")
        controller.alpha = 1.0
        with torch.no_grad():
            output = pipe(
                prompt=[prompt],
                num_inference_steps=args.num_inference_steps,
                num_waveforms_per_prompt=1,
                audio_start_in_s=0.0,
                audio_end_in_s=audio_len,
                generator=torch.Generator(device=device).manual_seed(int(seed)),
                steer=1,
                output_type="np",
            )
        edit_path = os.path.join(save_audio_dir, f"{prompt}_{j}_edited.wav")
        sf.write(edit_path, output.audios[0].T, sample_rate)
        logger.info(f"Saved to {edit_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--num_sample", type=int, default=3)
    parser.add_argument("--audio_len", type=float, default=None)
    parser.add_argument("--prompt", type=str, default="a music piece")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_last", action="store_true")
    args = parser.parse_args()
    main(args)