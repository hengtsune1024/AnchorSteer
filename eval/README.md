# Evaluation

Computes audio editing metrics for one or more methods against a task JSON benchmark (e.g. ZoME-Bench). Outputs a per-segment CSV and an aggregated summary CSV.

---

## Installation

```bash
pip install -r eval/requirements.txt
```

---

## CLAP Checkpoint

The evaluator uses [LAION-CLAP](https://huggingface.co/lukewys/laion_clap). The checkpoint (`music_audioset_epoch_15_esc_90.14.pt`) is **downloaded automatically** on first run and cached next to `eval.py`.

To use a pre-downloaded copy instead, set the environment variable:

```bash
export MELODIA_CLAP_CKPT=/path/to/music_audioset_epoch_15_esc_90.14.pt
```

---

## Task JSON Format

The `--task_json` file must contain two top-level keys:

```json
{
    "stats": {
        "<editing_type_id>": {
            "<attribute_name>": <any value>
        }
    },
    "segments": [
        {
            "ytid":             "video_id",
            "segment_id":       "seg000",
            "editing_type_id":  1,
            "wav":              "segments_47s/filename.wav",
            "emphasize":        "rock",
            "original_prompt":  "[piano] solo piece",
            "editing_prompt":   "a rock piece with electric guitar"
        }
    ]
}
```

| Field | Description |
|-------|-------------|
| `stats` | Keys are `editing_type_id` values; the attribute names inside are used to auto-infer whether a task is a **genre** or **instrument** edit, which controls how CLAP prompts are phrased. |
| `segments[].wav` | Path to the source WAV relative to `--src_root`. Leading `-` in the filename is automatically remapped to `_`. |
| `segments[].emphasize` | Target concept (e.g. `"rock"`, `"piano"`). |
| `segments[].original_prompt` | Source concept, read from the bracket term `[piano]` if present. Used to compute `src_CLAP_original` and `edit_CLAP_original`. |

---

## Edit File Matching

`--edit_name_mode` controls how edited WAV files are located inside `--method name=DIR`:

### `attr_suffix` (default for AnchorSteer)

The edited filename must contain the target attribute between the source stem and the method suffix:

```
<source_stem>_<attribute><suffix>.wav
<source_stem>-<attribute><suffix>.wav
```

Example — source `ABC_seg000.wav`, attribute `piano`, suffix `transformer`:

```
ABC_seg000_piano_transformer.wav   ✓
ABC_seg000_piano-transformer.wav   ✓
```

Multi-word attributes are tried in all normalised forms: `hip hop`, `hiphop`, `hip_hop`, `hip-hop`.

### `suffix`

The edited filename only adds a suffix to the source stem (no attribute token required):

```
<source_stem><suffix>.wav
```

Example — source `ABC_seg000.wav`, suffix `_anchosteer`:

```
ABC_seg000_anchorsteer.wav   ✓
```

---

## Usage

```bash
python eval/eval.py \
    --task_json   path/to/benchmark.json \
    --src_root    path/to/source_audio_root \
    --method      anchorsteer=path/to/edited_wavs \
    --edit_suffix transformer \
    --edit_name_mode attr_suffix \
    --out_csv     results/anchorsteer.csv
```

To compare multiple methods in one run, repeat `--method`:

```bash
python eval/eval.py \
    --task_json  path/to/benchmark.json \
    --src_root   path/to/source_audio_root \
    --method     anchorsteer=results/anchorsteer_edits \
    --method     baseline=results/baseline_edits \
    --edit_suffix transformer \
    --edit_name_mode attr_suffix \
    --out_csv    results/comparison.csv
```

---

## All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_json` | *(required)* | Path to the benchmark JSON. |
| `--src_root` | *(required)* | Root directory containing the source WAV files referenced in the JSON. |
| `--method` | *(required, repeatable)* | `name=DIR` pair. Repeat to evaluate multiple methods at once. |
| `--edit_suffix` | `_None` | Suffix appended to (or after the attribute in) the edited filename. |
| `--edit_name_mode` | `suffix` | `attr_suffix` (attribute must appear in filename) or `suffix` (suffix only). |
| `--mirror_subdir` | off | Preserve the source's parent subdirectory name when searching for edited WAVs. |
| `--out_csv` | *(required)* | Output path for the per-segment CSV. |
| `--out_summary_csv` | *(auto)* | Output path for the summary CSV. Defaults to `<out_csv stem>_summary.csv`. |
| `--device` | `cuda` | Torch device for CLAP and LPIPS. |
| `--clap_chunk_sec` | `10.0` | Audio is split into chunks of this length before CLAP embedding. |
| `--clap_prompt_reduce` | `max` | Aggregate CLAP scores over multiple prompts: `max` or `mean`. |
| `--clap_map01` | off | Map cosine similarity from `[−1, 1]` to `[0, 1]` before reporting. |
| `--lpaps_patch_sec` | `2.0` | Mel-spectrogram patch length (seconds) for LPIPS. |
| `--lpaps_hop_sec` | `2.0` | Hop between consecutive patches for LPIPS. |
| `--lpaps_reduce` | `sum` | Aggregate patch-level LPIPS scores: `sum` or `mean`. |
| `--use_json_editing_prompt` | off | Append `editing_prompt` from the JSON as an extra CLAP target prompt. |

---

## Metrics

All metrics are computed per source–edit pair and then averaged across segments in the summary CSV.

| Metric | Higher = | Description |
|--------|----------|-------------|
| `CLAP` | Better | CLAP similarity between the **edited** audio and target-concept prompts (same as `edit_CLAP_target`). Primary editing quality metric. |
| `src_CLAP_target` | — | CLAP of **source** audio vs. target-concept prompts. Baseline for how concept-related the source already is. |
| `edit_CLAP_target` | Better | CLAP of **edited** audio vs. target-concept prompts. |
| `src_CLAP_original` | — | CLAP of **source** audio vs. original-concept prompts. |
| `edit_CLAP_original` | Lower | CLAP of **edited** audio vs. original-concept prompts. Should decrease after successful editing. |
| `src_GAP` | — | `src_CLAP_target − src_CLAP_original`. Semantic gap before editing. |
| `edit_GAP` | Higher | `edit_CLAP_target − edit_CLAP_original`. Semantic gap after editing. |
| `GAP_delta` | Higher | `edit_GAP − src_GAP`. Net widening of the semantic gap due to editing. Primary combined metric. |
| `delta_target` | Higher | `edit_CLAP_target − src_CLAP_target`. How much the edit moved toward the target concept. |
| `delta_away_original` | Higher | `src_CLAP_original − edit_CLAP_original`. How much the edit moved away from the original concept. |
| `LPIPS` | Lower | Perceptual distance on log-mel spectrogram patches (VGG-LPIPS). Measures structural distortion. |
| `Chroma` | Higher | Fraction of frames where the dominant chroma pitch class matches the source. Measures melody preservation. |

---

## Output Files

**Per-segment CSV** (`--out_csv`): one row per (method, segment) pair with all metric values, the resolved source and edited WAV paths, and the CLAP prompts used.

**Summary CSV** (`--out_summary_csv`): one row per method with `mean` and `std` of every metric, plus `n_pairs`. Also printed to stdout at the end of the run.
