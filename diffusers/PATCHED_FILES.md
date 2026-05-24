# diffusers — Local Fork Notes

## Version

The `diffusers/` directory is a **verbatim copy** of `diffusers==0.37.0.dev0` as confirmed by `diffusers/src/diffusers/__init__.py`. No files within this fork have been modified.

## No source-level patches

No files within this fork were modified. The `concept` and `timestep_idx` injection is implemented entirely via Python subclassing in `models/stable_audio.py`:

- `MyStableAudioDiTModel` overrides `forward()` to accept and propagate the concept delta
- `MyStableAudioPipeline` overrides `__call__()` to pass controller outputs into the denoising loop

This subclassing approach means that the diffusers fork source files remain byte-for-byte identical to the upstream `0.37.0.dev0` snapshot. Concept injection does **not** require any edits to the fork.

## Why the fork is included

The fork is included to provide a **stable pinned version** (`0.37.0.dev0`) that:

1. Is not yet available as a stable PyPI release at the time of development.
2. Is compatible with the Stable Audio Open (SAO) model architecture used in AnchorSteer, specifically the `StableAudioDiTModel` and `StableAudioPipeline` classes that `MyStableAudioDiTModel` and `MyStableAudioPipeline` subclass.

Pinning the fork prevents silent breakage from upstream API changes.

## MuseControlLite note

The `models/musecontrollite/` directory contains `StableAudioDiTModel.py`, which is a **standalone variant** of the DiT model used exclusively by the MuseControlLite backbone (`MuseControlLite/MuseControlLite_setup.py`) when `use_concept=True`. It is a self-contained class that does not modify any files in this diffusers fork.

## Installation

Install the local fork in editable mode:

```bash
pip install -e ./diffusers
```

This must be done from the `AnchorSteer/` root directory (i.e., the parent of `diffusers/`).

## Warning: do NOT install from PyPI

**Do NOT run `pip install diffusers` from PyPI after performing the step above.**

Installing the PyPI `diffusers` package will silently overwrite the local fork in the Python environment. When this happens:

- `MyStableAudioDiTModel` and `MyStableAudioPipeline` may fail to import or behave incorrectly.
- Concept injection will silently produce unsteered outputs with no error.
- The startup guard in `models/utils.py` will detect the version mismatch and raise a `RuntimeError` with an explanatory message.

If you accidentally install from PyPI, re-run `pip install -e ./diffusers` from the `AnchorSteer/` root to restore the correct fork.

## Upgrading instructions

To upgrade to a newer upstream diffusers version:

1. Replace the entire `diffusers/` directory with a verbatim copy of the new upstream release.
2. Re-run `pip install -e ./diffusers`.
3. Verify that `MyStableAudioDiTModel.forward()` in `models/stable_audio.py` remains compatible with the new `StableAudioDiTModel.forward()` signature.
4. Verify that `MyStableAudioPipeline.__call__()` in `models/stable_audio.py` remains compatible with the new `StableAudioPipeline.__call__()` signature.
5. Run the test suite (`python test.py --exp_dir exps/<any-concept>`) and confirm that edited outputs are still concept-steered.

Update the version string in this file to reflect the new upstream version after a successful upgrade.
