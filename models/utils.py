import torch
import torch.nn.functional as F
try:
    import diffusers as _diffusers
    _ver = _diffusers.__version__
    if "dev" not in _ver:
        raise AssertionError
except (ImportError, AssertionError):
    raise RuntimeError(
        "The installed diffusers does not appear to be the patched local fork "
        f"(found version: {_ver if '_ver' in dir() else 'not installed'}). "
        "Run: pip install -e ./diffusers\n"
        "Do NOT install diffusers from PyPI afterward."
    )

def load_pipeline(
    pipe_type,
    model_id="stabilityai/stable-audio-open-1.0",
    dtype=torch.float16, 
    device=torch.device("cuda")
):
    if pipe_type == "interpret":
        from .stable_audio import MyStableAudioDiTModel, MyStableAudioPipeline
    else:
        raise ValueError(pipe_type)

    my_transformer = MyStableAudioDiTModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        torch_dtype=dtype
    )
    pipe = MyStableAudioPipeline.from_pretrained(
        model_id,
        transformer=my_transformer, 
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    return pipe


def slerp(h0, h1, alpha):
    """
    Spherical linear interpolation (slerp) over (B, L, D) tensors.
    h0: original hidden state (B, 1024, 1536)
    h1: edited hidden state (h + delta_h)
    alpha: interpolation coefficient (0.0 ~ 1.0)
    """
    # 1. Normalize to unit sphere
    h0_norm = F.normalize(h0, p=2, dim=-1)
    h1_norm = F.normalize(h1, p=2, dim=-1)

    # 2. Compute cosine angle (dot product)
    # Result shape: (B, 1024, 1)
    dot = torch.sum(h0_norm * h1_norm, dim=-1, keepdim=True)

    # Numerical stability: avoid nan in acos
    dot = torch.clamp(dot, -1.0, 1.0)

    # 3. Compute angle theta
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # 4. Handle near-zero angle (degenerate case: use linear interpolation)
    mask = sin_theta < 1e-6

    # 5. Slerp formula
    # res = [sin((1-a)*theta)/sin(theta)] * h0 + [sin(a*theta)/sin(theta)] * h1
    weight0 = torch.sin((1.0 - alpha) * theta) / (sin_theta + 1e-8)
    weight1 = torch.sin(alpha * theta) / (sin_theta + 1e-8)

    # For near-zero angles, fall back to linear weights
    weight0[mask] = 1.0 - alpha
    weight1[mask] = alpha

    h_interp = weight0 * h0 + weight1 * h1

    return h_interp

def norm_fixed_interpolation(h, delta_h, alpha):
    """
    Norm-corrected linear interpolation between h and (h + delta_h).
    """
    h_target = h + delta_h

    h_interp = (1.0 - alpha) * h + alpha * h_target

    orig_norm = torch.norm(h, dim=-1, keepdim=True)
    target_norm = torch.norm(h_target, dim=-1, keepdim=True)

    # Compute the desired interpolated norm
    desired_norm = (1.0 - alpha) * orig_norm + alpha * target_norm

    # Rescale the interpolated vector to the desired norm
    current_norm = torch.norm(h_interp, dim=-1, keepdim=True)
    h_final = h_interp * (desired_norm / (current_norm + 1e-8))

    return h_final