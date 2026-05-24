# Evaluation

This folder contains the evaluation code for edited music.

The evaluator computes:

- `CLAP`
- `LPIPS`
- `Chroma`
- source/edit CLAP scores for the original and target attributes

## CLAP checkpoint

Please manually download `music_audioset_epoch_15_esc_90.14.pt` from [`lukewys/laion_clap`](https://huggingface.co/lukewys/laion_clap) and place it in the same directory as `eval.py`.


## How To Run

run one method directly:

```bash
python eval.py \
  --task_json "$TASK_JSON" \
  --src_root "$SRC_ROOT" \
  --method "$METHOD_NAME=$EDIT_DIR" \
  --edit_name_mode "$EDIT_NAME_MODE" \
  --edit_suffix "$EDIT_SUFFIX" \
  --out_csv "$OUT_CSV" \
  --lpaps_reduce sum \
  --lpaps_patch_sec 2 \
  --lpaps_hop_sec 2 \
  --clap_chunk_sec 10
```

## Edit File Matching

These two args control how source wavs are matched to edited wavs:

```bash
--edit_name_mode attr_suffix
--edit_suffix vector
```

With `attr_suffix`, the script expects the edited filename to contain:

```text
source_stem + target_attribute + edit_suffix + .wav
```

For example, if the source file is:

```text
ABC_seg000.wav
```

and the JSON target attribute is:

```text
piano
```

then:

```bash
--edit_name_mode attr_suffix
--edit_suffix vector
```

will try names like:

```text
ABC_seg000_piano_vector.wav
ABC_seg000_piano-vector.wav
```

The script also tries small filename variants, such as:

```text
vector
_vector
-vector
```

Use `--edit_name_mode suffix` instead if edited files only add a suffix to the source stem, for example:

```text
ABC_seg000_vector.wav
```

 


