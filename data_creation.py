import os
import json
from tqdm import trange
import torch 
from diffusers import StableAudioPipeline
from dataclasses import dataclass
from typing import List
import soundfile as sf
import argparse
from misc import set_seed

def build_pipe():
    # Use float16 only when CUDA is available. Avoid loading float16 models on CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        # Do not request float16 on CPU (unsupported); use default dtype (float32).
        pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0")
        pipe = pipe.to("cpu")
        print("Warning: CUDA not available — running pipeline on CPU with float32. For better performance, use an accelerator and float16.")
    return pipe

def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]

class DataCreator:
    def __init__(self, cfg, pipe):
        self.pipe = pipe
        self.root_dir = cfg.root_dir
        self.batch_size = cfg.batch_size
        self.num_samples = cfg.num_samples
        self.reference_prompt = cfg.reference_prompt
        self.audio_len = cfg.audio_len if cfg.audio_len > 0 else None
        self.training_labels = repeat_ntimes(cfg.training_labels, cfg.num_samples)

        print(f"to create {self.num_samples} total number of samples in {cfg.root_dir}")

    def create_images(self, num_inference_steps=50):
        pipe = self.pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)

        os.makedirs(self.root_dir, exist_ok=True)
        if all(os.path.exists(os.path.join(self.root_dir, f"{j:04d}.wav")) for j in range(self.num_samples)):
            return
        
        for idx in trange(0, self.num_samples, self.batch_size):

            n = min(idx+self.batch_size, self.num_samples) - idx
            prompts = [self.reference_prompt[0]] * n
            negative_prompts = [self.reference_prompt[1]]*n if len(self.reference_prompt) == 2 else None

            fnames = [os.path.join(self.root_dir, f"{idx+j:04d}.wav") for j in range(n)]

            with torch.no_grad(), torch.inference_mode():
                output = pipe(
                    prompts,
                    negative_prompt=negative_prompts,
                    num_inference_steps=num_inference_steps,
                    audio_start_in_s=0.0,
                    audio_end_in_s=self.audio_len,
                    output_type="np",
                ).audios
            for audio, fname in zip(output, fnames):
                sf.write(fname, audio.T, pipe.vae.sampling_rate)
            torch.cuda.empty_cache()

    def create_labels(self):
        os.makedirs(self.root_dir, exist_ok=True)
        json.dump(self.training_labels, open(self.root_dir + "/labels.json", "w"))
        prompt = self.training_labels[0][0][0]
        json.dump(prompt, open(self.root_dir + "/test.json", "w"))

@dataclass
class Cfg:
    root_dir: str
    num_samples: int
    audio_len: float
    batch_size: int = 2
    reference_prompt: List[str] = None
    training_labels: list = None

def main(args):

    pipe = build_pipe()

    with open(args.config, "r") as f:
        config = json.load(f)
    
    cfg = Cfg(
        root_dir=config["root_dir"],
        num_samples=config["num_samples"],
        audio_len=config["audio_len"],
        batch_size=args.batch_size,
        reference_prompt=config["reference_prompt"],
        training_labels=config["training_labels"],
    )
    print(cfg)

    set_seed(111)
    creator=DataCreator(cfg, pipe)
    creator.create_labels()
    creator.create_images()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    main(args)
