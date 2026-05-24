import glob
import os
import json
import torch
from tqdm import trange
import torchaudio

def prepare_model_inputs(waveforms, prompts, pipe):
    # 1. Prepare Inputs
    waveforms = waveforms.to(device=pipe.device, dtype=pipe.vae.dtype)
    
    # Check dimensions: [batch, time] or [batch, 1, time]
    if waveforms.ndim == 2:
        waveforms = waveforms.unsqueeze(1)
    
    with torch.no_grad():

        encoder_outputs = pipe.vae.encode(waveforms)
        latents = encoder_outputs.latent_dist.sample()
        bsz = latents.shape[0]
        
        prompt_embeds = pipe.encode_prompt(
            prompt=prompts,
            device=pipe.device,
            do_classifier_free_guidance=False # 訓練時通常只計算 Conditional，除非要做 CFG Distillation
        )
        
        # (B) Encode Duration (Start/End Seconds)
        # 計算音訊秒數
        audio_length_samples = waveforms.shape[-1]
        audio_seconds = audio_length_samples / pipe.vae.sampling_rate

        # 建構 seconds_start (0s) 和 seconds_end (總長度)
        start_seconds = [0.0] * bsz
        end_seconds = [audio_seconds] * bsz
        
        seconds_start_hidden_states, seconds_end_hidden_states = pipe.encode_duration(
            start_seconds, 
            end_seconds, 
            device=pipe.device, 
            do_classifier_free_guidance=False,
            batch_size=bsz
        )

        # (C) Concatenate Embeddings (依照 Pipeline 邏輯)
        # 序列特徵: Text + Start + End
        text_audio_duration_embeds = torch.cat(
            [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
        )
        # 全域特徵: Start + End
        audio_duration_embeds = torch.cat(
            [seconds_start_hidden_states, seconds_end_hidden_states], dim=2
        )
    torch.cuda.empty_cache()
    return latents.cpu(), text_audio_duration_embeds.cpu(), audio_duration_embeds.cpu()

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, audio_folder, pipe, encode_batch_size=4):
        audio_paths = sorted(glob.glob(os.path.join(audio_folder, "*.wav")))
        with open(os.path.join(audio_folder, 'labels.json'), 'r') as f:
            labels = json.load(f)

        self.latents = []
        self.text_audio_duration_embeds = []
        self.audio_duration_embeds = []

        N = len(audio_paths)
        for i in trange(0, N, encode_batch_size):
            curr_batch = min(i+encode_batch_size, N) - i
            wavs = []
            for j in range(i, i+curr_batch):
                wav, sr = torchaudio.load(audio_paths[j])
                wavs.append(wav)
            wavs = torch.stack(wavs, dim=0)

            prompts = [l[0][0] for l in labels[i: i+curr_batch]]

            latents, text_audio_duration_embeds, audio_duration_embeds = prepare_model_inputs(wavs, prompts, pipe)

            self.latents.append(latents)
            self.text_audio_duration_embeds.append(text_audio_duration_embeds)
            self.audio_duration_embeds.append(audio_duration_embeds)
            
        self.latents = torch.cat(self.latents, dim=0)
        self.text_audio_duration_embeds = torch.cat(self.text_audio_duration_embeds, dim=0)
        self.audio_duration_embeds = torch.cat(self.audio_duration_embeds, dim=0)

    def __getitem__(self, idx):
        return self.latents[idx], self.text_audio_duration_embeds[idx], self.audio_duration_embeds[idx]

    def __len__(self):
        return len(self.latents)


def get_dataloader(audio_folder, pipe, mini_batch_size, encode_batch_size=4, num_workers=4, shuffle=False):
    dataset = TrainingDataset(audio_folder, pipe, encode_batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=mini_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader



