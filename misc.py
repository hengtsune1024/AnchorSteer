import os
import torch
import numpy as np
import soundfile as sf
import random

def save_wav(path: str, audio: np.ndarray, sample_rate: int):
    audio = np.asarray(audio)
    audio = np.nan_to_num(audio)
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio_int16, samplerate=sample_rate)

def set_seed(seed_value=42):
    """Set seeds for reproducibility."""
    random.seed(seed_value)                # Python random seed
    np.random.seed(seed_value)             # NumPy random seed
    torch.manual_seed(seed_value)          # PyTorch CPU and CUDA seed (sets for all devices)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)       # Set seed for current GPU
        torch.cuda.manual_seed_all(seed_value)   # Set seed for all GPUs if using multi-GPU
        
        # Optional: ensure deterministic behavior for some cuDNN operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

