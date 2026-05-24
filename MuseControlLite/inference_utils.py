import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
)
import os
from MuseControlLite.config_inference import get_config
from models import Controller, load_pipeline

class Inference:
    def __init__(self, use_best=True):
        self.config = get_config()
        self.weight_dtype = torch.float16 if self.config["weight_dtype"] == "fp16" else torch.float32

        # determine device from availability and config (allow '-1' to force CPU)
        use_cuda = torch.cuda.is_available() and str(self.config.get("GPU_id", "0")) != "-1"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.use_best = use_best

        self.condition_extractors = None
        self.pipe = None
        self._load_pipeline(self.config)

        # cache controller state_dicts on CPU to save GPU memory
        self.controller = None
        self.concept_dir = None

    def _load_pipeline(self, config):
        if config["apadapter"]:
            self.condition_extractors, transformer_ckpt = initialize_condition_extractors(config)
            pipe = setup_MuseControlLite(config, self.weight_dtype, transformer_ckpt, use_concept=True)
            print(f"Using pipeline MuseControlLite")
        else:
            pipe = load_pipeline("interpret", dtype=self.weight_dtype)
            print(f"Using pipeline Stable Audio")
        pipe = pipe.to(self.device)
        self.pipe = pipe
        self.apadapter = self.config["apadapter"]

    def _load_controller(self, concept_dir=None):
        # If disabling, remove controller from pipe and free GPU memory
        if concept_dir is None:
            print("Disabling controller")
            if self.controller is not None:
                self.pipe.set_controller(None)
                self.controller.to("cpu")
                del self.controller
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            self.controller = None
            self.concept_dir = None
            return

        print(f"Using controller from {concept_dir}")
        if concept_dir == self.concept_dir:
            return

        yaml = YAML()
        with open(os.path.join(concept_dir, 'config.yaml'), "r", encoding="utf-8") as f:
            concept_configs = yaml.load(f)
        num_hidden_vec = (1024 if concept_configs["train_full_len"] else 107) + \
                        (1 if concept_configs.get("inject_first_state", False) else 0)
        model_path = os.path.join(concept_dir, 'best.pth' if self.use_best else 'adaptor.pth')
        state = torch.load(model_path, map_location="cpu")

        # free previous active controller from GPU before switching
        if self.controller is not None:
            self.pipe.set_controller(None)
            self.controller.to("cpu")
            del self.controller
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # build controller instance and load cached state, then move to device
        controller = Controller(
            edit_layers=list(range(concept_configs["edit_start"], concept_configs["edit_end"]+1)),
            use_time=concept_configs["use_time_emb"],
            total_layers=len(self.pipe.transformer.transformer_blocks),
            control_type=concept_configs.get("control_type", "vector"),
            inject_first_state=concept_configs.get("inject_first_state", False),
            hidden_dim=(num_hidden_vec, 1536)
        )
        try:
            controller.load_state_dict(state)
        except Exception:
            controller.load_state_dict(state, strict=False)
        controller.eval()
        controller = controller.to(self.device)

        self.controller = controller
        self.concept_dir = concept_dir
        self.pipe.set_controller(self.controller)

    def infer_single(
        self, 
        wav_path, 
        prompt, 
        output_dir, 
        negative_prompt="", 
        concept_dir=None, 
        seed=0, 
        num_steps=50, 
        skip_existed=False
    ):
        os.makedirs(output_dir, exist_ok=True)
        output_name = f"{os.path.splitext(os.path.basename(wav_path))[0]}_{os.path.basename(concept_dir) if concept_dir else 'None'}.wav"
        gen_file_path = os.path.join(output_dir, output_name)
        if skip_existed and os.path.exists(gen_file_path):
            print(f"{wav_path} is already processed : {gen_file_path}")
            return gen_file_path

        print(f"Processing {wav_path}")
        with open(os.path.join(output_dir, "description.txt"), 'a') as file:
            file.write(f'Prompt: {prompt}, File: {wav_path}, Concept: {concept_dir}, Seed: {seed}\n')

        # load controller
        self._load_controller(concept_dir)
        concept_input = 1 if concept_dir is not None else 0


        pipe_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_steps,
            "num_waveforms_per_prompt": 1,
            "audio_end_in_s": None,
            "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
            "output_type": "np"
        }

        if concept_dir is not None:
            pipe_kwargs["steer"] = concept_input

        if self.apadapter:
            final_condition, final_condition_audio = process_musical_conditions(self.config, wav_path, self.condition_extractors, output_dir, 0, self.weight_dtype, self.pipe)
            pipe_kwargs["extracted_condition"] = final_condition
            pipe_kwargs["extracted_condition_audio"] = final_condition_audio
            pipe_kwargs["guidance_scale_text"] = self.config["guidance_scale_text"]
            pipe_kwargs["guidance_scale_con"] = self.config["guidance_scale_con"]
            pipe_kwargs["guidance_scale_audio"] = self.config["guidance_scale_audio"]
        else:
            pipe_kwargs["guidance_scale"] = self.config["guidance_scale_text"]
        
        with torch.no_grad():
            waveform = self.pipe(**pipe_kwargs).audios
        
        waveform = waveform[0].T
        info = sf.info(wav_path)
        if waveform.shape[0] > info.frames:
            waveform = waveform[:info.frames, :]

        sf.write(gen_file_path, waveform, self.pipe.vae.sampling_rate)

        orig_path = os.path.join(output_dir, f"original_0.wav")
        if os.path.exists(orig_path):
            os.remove(orig_path)

        if self.config['show_result_and_plt']:
            evaluate_and_plot_results(wav_path, gen_file_path, output_dir, 0)
        return gen_file_path
