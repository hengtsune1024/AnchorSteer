import os
import sys
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
import soundfile as sf
from ruamel.yaml import YAML
from MuseControlLite.MuseControlLite_setup import (
    setup_MuseControlLite,
    initialize_condition_extractors,
    evaluate_and_plot_results,
    process_musical_conditions,
    build_inputs
)
import os
import numpy as np
from MuseControlLite.config_inference import get_config
from models import Controller, load_pipeline
import argparse
import json

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if config["weight_dtype"] == "fp16":
        weight_dtype = torch.float16
    if config["apadapter"]:
        condition_extractors, transformer_ckpt = initialize_condition_extractors(config)
        pipe = setup_MuseControlLite(config, weight_dtype, transformer_ckpt, use_concept=(len(config["concept_cond"])>0))
        pipe = pipe.to("cuda")
    else:
        pipe = load_pipeline("interpret", dtype=weight_dtype)
    # add concept controller
    if config["concept_cond"]:
        print(f"Using controller from {config['concept_cond']}")
        yaml = YAML()
        with open(os.path.join(config["concept_cond"], 'config.yaml'), "r", encoding="utf-8") as f:
            concept_configs = yaml.load(f)
        num_hidden_vec = (1024 if concept_configs["train_full_len"] else 107) + \
                         (1 if concept_configs.get("inject_first_state", False) else 0)
        controller = Controller(
            edit_layers=list(range(concept_configs["edit_start"], concept_configs["edit_end"]+1)),
            use_time=concept_configs["use_time_emb"],
            total_layers=len(pipe.transformer.transformer_blocks),
            control_type=concept_configs.get("control_type", "vector"),
            inject_first_state=concept_configs.get("inject_first_state", False),
            hidden_dim=(num_hidden_vec, 1536)
        )
        model_path = os.path.join(config["concept_cond"], 'best.pth')
        controller.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
        controller.eval()
        controller = controller.to("cuda")
        pipe.set_controller(controller)

    prompts, audio_files, concept_inputs, seeds = build_inputs(config)
    N = len(prompts)
    cond_cache = {}

    negative_text_prompt = config["negative_text_prompt"]
    # Apply masks for audio condition and musical attribute condition, the masked parts will be assign to zero, sames are the drop condition in cfg.
    score_dynamics = []
    score_rhythm = []
    score_melody = []
    with torch.no_grad():
        for i in range(N):
            seed = seeds[i]
            prompt_texts = prompts[i]
            audio_file = audio_files[i]
            concept = concept_inputs[i]

            description_path = os.path.join(output_dir, "description.txt")
            with open(description_path, 'a') as file:
                file.write(f'Prompt: {prompt_texts}, File: {audio_file}, Concept: {concept}, Seed: {seed}\n')
            
            info = sf.info(audio_file)
            duration = info.frames / info.samplerate
            pipe_kwargs = {
                "prompt": prompt_texts,
                "negative_prompt": negative_text_prompt,
                "num_inference_steps": config["denoise_step"],
                "num_waveforms_per_prompt": 1,
                "audio_end_in_s": duration,
                "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
                "output_type": "np"
            }

            if config['concept_cond']:
                pipe_kwargs["steer"] = concept

            if config["apadapter"]:
                if audio_file not in cond_cache:
                    final_condition, final_condition_audio = process_musical_conditions(config, audio_file, condition_extractors, output_dir, i, weight_dtype, pipe)
                    cond_cache[audio_file] = (final_condition, final_condition_audio)
                else:
                    final_condition, final_condition_audio = cond_cache[audio_file]
                pipe_kwargs["extracted_condition"] = final_condition
                pipe_kwargs["extracted_condition_audio"] = final_condition_audio 
                pipe_kwargs["guidance_scale_text"] = config["guidance_scale_text"]
                pipe_kwargs["guidance_scale_con"] = config["guidance_scale_con"]
                pipe_kwargs["guidance_scale_audio"] = config["guidance_scale_audio"]
            else:
                pipe_kwargs["guidance_scale"] = config["guidance_scale_text"]

            waveform = pipe(**pipe_kwargs).audios
            gen_file_path = os.path.join(output_dir, f"{'muse' if config['apadapter'] else 'stable'}_{prompt_texts}_{'orig' if concept == 0 else 'edit'}_{i}.wav")
            sf.write(gen_file_path, waveform[0].T, pipe.vae.sampling_rate)
            print(f"Saved to {gen_file_path}")

            orig_path = os.path.join(output_dir, f"original_{i}.wav")
            if os.path.exists(orig_path):
                os.remove(orig_path)

            if config['show_result_and_plt']:
                dynamics_score, rhythm_score, melody_score = evaluate_and_plot_results(
                    audio_file, gen_file_path, output_dir, i
                )
                score_dynamics.append(dynamics_score)
                score_rhythm.append(rhythm_score)
                score_melody.append(melody_score)   

        data_to_save = {"config": config}
        if config['show_result_and_plt']:
            data_to_save["score_dynamics"] = np.mean(score_dynamics)
            data_to_save["score_rhythm"] = np.mean(score_rhythm)
            data_to_save["score_melody"] = np.mean(score_melody)
        file_path = os.path.join(output_dir, "result.txt")
        with open(file_path, "w") as file:
            json.dump(data_to_save, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    parser.add_argument("--concept_cond", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--apadapter", type=int, default=None)
    args = parser.parse_args()

    config = get_config()  # Pass the parsed arguments to get_config
    if args.audio_dir is not None:
        config["audio_files"] = [os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) if f.startswith("src_")]
        print("Audio files: ", len(config["audio_files"]))
    if args.concept_cond is not None:
        config["concept_cond"] = args.concept_cond
    if args.text is not None:
        config["text"] = [args.text] * len(config["audio_files"])
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.apadapter is not None:
        config["apadapter"] = args.apadapter == 1
    main(config)
