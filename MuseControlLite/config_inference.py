def get_config():
    return {
        "condition_type": ["dynamics", "rhythm", "melody_mono"], #  you can choose any combinations in the two sets: ["dynamics", "rhythm", "melody_mono", "audio"],  ["melody_stereo", "audio"]
                                    # When using audio, is recommend to use empty string "" as prompt
        "output_dir": "./generated_audio/output",

        "GPU_id": "0",

        "apadapter": True, # True for MuseControlLite, False for original Stable-audio

        "ap_scale": 1.0, # recommend 1.0 for MuseControlLite, other values are not tested

        "guidance_scale_text": 10.0,

        "guidance_scale_con": 1.0, # The separated guidance for Musical attribute condition
        
        "guidance_scale_audio": 1.0,
        
        "denoise_step": 50,

        "sigma_min": 0.3, # sigma_min and sigma_max are for the scheduler.

        "sigma_max": 500,  # Note that if sigma_max is too large or too small, the "audio condition generation" will be bad.

        "weight_dtype": "fp32", # fp16 and fp32 sounds quiet the same.

        "negative_text_prompt": "",

        ###############

        "audio_mask_start_seconds": 14, # Apply mask to musical attributes choose only one mask to use, it automatically generates a complemetary mask to the other condition

        "audio_mask_end_seconds": 47, 

        "musical_attribute_mask_start_seconds": 0, # 'Apply mask to audio condition, choose only one mask to use, it automatically generates a complemetary mask to the other condition'

        "musical_attribute_mask_end_seconds": 0,

        ###############

        "no_text": False, # Optional, set to true if no text prompt is needed (possible for audio inpainting or outpainting)

        "show_result_and_plt": False,

        "audio_files": ["generated_audio/cond/stable_a music piece_orig_0.wav"],

        "text": [""],

        "concept_cond": "exps/_eval/piano-vector",
        # "concept_cond": "",

        ########## adapters avilable ############
        # We trained 4 set of adapters:
        # 1. with conditions ["melody_mono", "dynamics", "rhythm"]
        # 2. with conditions ["melody_mono"]
        # 3. with conditions ["melody_stereo"]
        # 3. with conditions ["audio"]
        # MuseControlLite_inference_all.py will automaticaly choose the most suitable model according to the condition type:
        ###############
        # Works for condition ["dynamics", "rhythm", "melody_mono"]
        "transformer_ckpt_musical": "MuseControlLite/checkpoints/woSDD-all/model_3.safetensors",
        
        "extractor_ckpt_musical": {
            "dynamics": "MuseControlLite/checkpoints/woSDD-all/model_1.safetensors",
            "melody": "MuseControlLite/checkpoints/woSDD-all/model.safetensors",
            "rhythm": "MuseControlLite/checkpoints/woSDD-all/model_2.safetensors",
        },
        ###############

        # Works for ['audio], it works without a feature extractor, and could cooperate with other adapters
        #################
        "audio_transformer_ckpt": "MuseControlLite/checkpoints/70000_Audio/model.safetensors",

        # Specialized for ['melody_stereo']
        ###############
        "transformer_ckpt_melody_stero": "MuseControlLite/checkpoints/70000_Melody_stereo/model_1.safetensors",

        "extractor_ckpt_melody_stero": {
            "melody": "MuseControlLite/checkpoints/70000_Melody_stereo/model.safetensors",
        },
        ###############

        # Specialized for ['melody_mono']
        ###############
        "transformer_ckpt_melody_mono": "MuseControlLite/checkpoints/40000_Melody_mono/model_1.safetensors",

        "extractor_ckpt_melody_mono": {
            "melody": "MuseControlLite/checkpoints/40000_Melody_mono/model.safetensors",
        },
        ###############
    }