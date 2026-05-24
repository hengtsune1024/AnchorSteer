# AnchorSteer

**AnchorSteer: Self-Discovered Concept Injection for Structure-Preserving Music Editing**

[[arXiv TBD]](arXiv TBD) | [[KDD '26 TBD]](KDD '26 TBD) | [[Demo TBD]](Demo TBD)

---

## Abstract

Controllable music editing is to modify high-level attributes while strictly preserving rhythmic and melodic structures. However, this task is challenged by a semantic-structural entanglement: steering methods often degrade structure to achieve editing performance, while structural adaptors suppress semantic responsiveness. We propose **AnchorSteer**, a framework that disentangles this tension by coupling structural anchoring with self-discovered semantic steering. The proposed approach probes internal representations to extract interpretable, label-free concept vectors via a self-supervised reconstruction objective, isolating attributes without curated data. During editing, these portable, plug-and-play concept vectors are injected into diffusion hidden manifolds while a structural adaptor enforces consistency. Variants for unconditioned and conditioned injections are provided to balance robustness and semantic strength. Experiments on ZoME-Bench and subjective tests show that the proposed framework outperforms both steering-only and anchoring-only baselines, enabling significant semantic transformations with high-fidelity structural preservation.

---

## Installation

### Hardware requirements

- Python 3.10
- CUDA 12.x (tested on 12.1 and 12.8)
- Single NVIDIA GPU with at least **16 GB VRAM** for training; **12 GB** is sufficient for inference only
- Reference hardware: RTX 4080 (16 GB) — approximately **30 minutes per concept** for 20 epochs

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yaml
conda activate env_anchorsteer
```

### 2. Install the patched local diffusers fork

```bash
pip install -e ./diffusers
```

> **WARNING:** Do not run `pip install diffusers` afterward; this will overwrite the patched local fork and cause concept injection to silently fail. See `diffusers/PATCHED_FILES.md` for details.

### 3. Log in to HuggingFace

```bash
hf auth login
```

Before running any script you must accept the Stability AI license agreement on the model page at [https://huggingface.co/stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0). The Stable Audio Open weights are downloaded automatically from HuggingFace on the first run and cached at `~/.cache/huggingface/`.

### 4. Download MuseControlLite checkpoints

The structure-preserving editing pipeline (`edit.py`) requires MuseControlLite adapter checkpoints. Download them from the [MuseControlLite project](https://github.com/fundwotsai2001/MuseControlLite):

```bash
# Run from the AnchorSteer root directory
gdown --folder 1Q9B333jcq1czA11JKTbM-DHANJ8YqGbP -O MuseControlLite/checkpoints
```

This step is only required for `edit.py`. Plain concept steering via `generate.py` does not use these checkpoints.

---

## Quick start: Steering-only inference

### 1. Download pre-trained concept checkpoints

> Download link: `[TBD]`
>
> ```bash
> gdown [TBD]
> ```

### 2. Extract to `exps/`

```bash
unzip checkpoints.zip -d exps/
```

### 3. Run inference

```bash
python generate.py --exp_dir exps/rock-transformer
```

Defaults: 3 samples, prompt `"a music piece"`. Override with `--num_sample N` and `--prompt "your prompt"`.

### 4. Expected output

Six WAV files will be written to `exps/rock-transformer/best/`:
- `a music piece_0_orig.wav` … `a music piece_2_orig.wav` — original (unsteered) generations
- `a music piece_0_edited.wav` … `a music piece_2_edited.wav` — concept-steered generations


---

## Quick start: AnchorSteer inference with MuseControlLite

To apply concept steering to an existing audio file while preserving its musical structure, use `edit.py`:

```bash
python edit.py \
    --source_audio path/to/source.wav \
    --concept_dir exps/rock-transformer
```

> **Prerequisites:** complete Installation steps 1–4, including the MuseControlLite checkpoint download.

The edited WAV is written to `outputs/<source_stem>_rock-transformer.wav`.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | `"a music piece"` | Text prompt describing the desired output |
| `--output_dir` | `outputs` | Directory where the edited WAV is saved |
| `--negative_prompt` | `""` | Negative text prompt |
| `--num_steps` | `50` | Number of diffusion sampling steps |
| `--seed` | `0` | Random seed for reproducibility |
| `--use_last` | off | Use `adaptor.pth` (final epoch) instead of `best.pth` |

---

## Training a custom concept

The following example trains a `rock` Controller from scratch. Replace `rock` with any concept name.

### Step 1: Create a dataset config

Create `dataset_config/rock.json` with the following schema:

```json
{
    "root_dir": "datasets/rock",
    "num_samples": 1000,
    "audio_len": -1,
    "reference_prompt": [
        "rock music with electric guitar and heavy drums",
        "Low quality"
    ],
    "training_labels": [
        [["a music piece", ["rock"]]]
    ]
}
```

#### Field reference

| Field | Type | Description |
|-------|------|-------------|
| `root_dir` | `string` | Directory where generated WAV files are saved (relative to the AnchorSteer root). |
| `num_samples` | `int` | Number of reference audio clips to generate. 1000 is recommended for full training; 10–50 is enough for a quick smoke test. |
| `audio_len` | `float` | Duration in seconds of each generated clip. `-1` generates the full 47-second audio (pair with `--train_full_len` during training). A positive value such as `5` produces shorter clips that train faster but capture less temporal structure. |
| `reference_prompt` | `[positive, negative]` | The prompt pair passed to Stable Audio Open when **generating** the reference audio. The positive prompt steers the style toward the target concept; the negative prompt suppresses unwanted qualities (e.g. `"Low quality"`). |
| `training_labels` | `list` | Maps each audio file to the (prompt, concept) pair used during **Controller training**. Format: `[[["text_prompt", ["concept_name"]]]]`. The text prompt is the neutral conditioning text the Controller sees at training time — keep it generic (e.g. `"a music piece"`) so the Controller learns a concept delta that is independent of any particular text input. |

> **`reference_prompt` vs `training_labels` text prompt:** these serve different roles. `reference_prompt` controls what audio gets generated (concept-specific, e.g. `"rock music with electric guitar"`). The text prompt inside `training_labels` is what the Controller is conditioned on during training (concept-neutral, e.g. `"a music piece"`). Keeping them separate prevents the Controller from entangling concept knowledge with text-prompt knowledge.

### Step 2: Generate reference audio

```bash
python data_creation.py --config dataset_config/rock.json
```

Expected output: `datasets/rock/0000.wav` through `datasets/rock/0999.wav` plus `labels.json` and `test.json`.

### Step 3: Train the Controller

```bash
python train.py \
    --train_data_dir datasets/rock \
    --output_dir exps/rock-transformer \
    --train_full_len
```

Expected training time: approximately **30 minutes** on an RTX 4080.

Expected outputs in `exps/rock-transformer/`:
- `best.pth` — best checkpoint (lowest validation loss)
- `adaptor.pth` — final-epoch checkpoint
- `config.yaml` — full argument snapshot for reproducibility
- `loss_history.png` — training / validation loss curves

---

## Key flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--control_type` | `transformer` | Controller architecture: `transformer` (bottleneck TransformerEncoder capturing temporal context) or `vector` (static learnable delta, ~1–3 M params, faster) |
| `--train_full_len` | `False` | Train on full 47-second audio; omit to use 5-second crops |
| `--edit_start` | `0` | Index of the first SAO DiT transformer block to inject the Controller delta |
| `--edit_end` | `23` | Index of the last SAO DiT transformer block to inject the Controller delta (inclusive) |
| `--learning_rate` | `5e-4` | AdamW learning rate |
| `--num_train_epochs` | `20` | Number of training epochs |
| `--mini_batch_size` | `2` | Mini-batch size per device; effective batch size = `mini_batch_size × gradient_accumulation_steps` |
| `--encode_batch_size` | `4` | Batch size for VAE-encoding the dataset during preprocessing; increase for faster encoding on high-VRAM GPUs |
| `--gradient_accumulation_steps` | `2` | Number of gradient accumulation steps before an optimizer step |

---

## Evaluation

See [`eval/README.md`](eval/README.md) for setup and usage. The evaluator computes CLAP, LPIPS, and Chroma scores for original and edited audio.

---

## Repository layout

```
AnchorSteer/
├── README.md
├── LICENSE
├── environment.yaml
├── requirements.txt
├── diffusers/              # Pinned local fork of huggingface/diffusers (0.37.0.dev0)
├── dataset_config/         # Concept JSON configs
├── eval/                   # Evaluation scripts (see eval/README.md)
├── models/                 # AnchorSteer core modules
│   ├── __init__.py
│   ├── controller.py       # Vector / Transformer Controller
│   ├── stable_audio.py     # MyStableAudioPipeline + MyStableAudioDiTModel
│   ├── utils.py            # load_pipeline() helper
│   └── musecontrollite/    # AnchorSteer-MuseControlLite integration layer
├── MuseControlLite/        # fundwotsai2001/MuseControlLite — structure-anchoring backbone
│   └── checkpoints/        # MuseControlLite adapter weights (download separately, see Installation)
├── data_creation.py        # Generate reference audio dataset for a concept
├── train.py                # Train Controller
├── generate.py             # Inference: generate original + concept-steered audio pairs from a text prompt
├── edit.py                 # Inference: structure-preserving concept editing of a source audio file
├── config.py               # Argument parser for train.py
├── utils_data.py           # DataLoader for pre-encoded latents
├── misc.py                 # set_seed and small utilities
└── doc/                    # Internal planning and change-log documents
```

---

## Acknowledgements

We thank the authors of the following open-source projects:

- **[InterpretDiffusion](https://github.com/hangligit/InterpretDiffusion)**
- **[MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite)**
- **[HuggingFace diffusers](https://github.com/huggingface/diffusers)**

---

## Citation

If you use AnchorSteer in your research, please cite:

```bibtex
@inproceedings{anchosteer2026,
  title     = {AnchorSteer: Self-Discovered Concept Injection for Structure-Preserving Music Editing},
  author    = {Chih-Heng Chang, Keng-Seng Ho, Chih-Yu Tsai, Kuan-Lin Chen, Yi-Hsuan Yang, Jian-Jiun Ding},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  year      = {2026},
  note      = {[arXiv TBD]}
}
```

---

## License

This project is released under the MIT License (see `LICENSE`). The MuseControlLite component (`MuseControlLite/`) is included under its own license; see `MuseControlLite/LICENSE` for details.
