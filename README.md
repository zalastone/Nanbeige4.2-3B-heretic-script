# Nanbeige 4.2 3B — Heretic script

This repository contains the patches and helper scripts needed to run
[Heretic](https://github.com/p-e-w/heretic) on the
[Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) model.

Nanbeige ships custom modeling code (`trust_remote_code=True`) and its code
targets an older `transformers` release. The patches below make it load and
run under `transformers` 5.x and Heretic 2.1.0.

## Usage

1. Download the model to a local directory, e.g. `.\Nanbeige4.2-3B\`.
2. Install Heretic:
   ```
   pip install heretic-llm
   ```
3. Apply the patches:
   ```
   python patch_heretic.py
   python patch_model.py --model-dir .\Nanbeige4.2-3B
   ```
4. Run Heretic in 4-bit (recommended on 8 GB VRAM):
   ```
   heretic --quantization bnb_4bit .\Nanbeige4.2-3B\
   ```

## Files

- `patch_heretic.py` — makes Heretic pass `trust_remote_code=True`.
- `patch_model.py` — updates Nanbeige's custom modeling code for transformers 5.x.
- `modeling_nanbeige.py` — the fully patched modeling file (drop-in replacement).
- `run.bat` — convenience launcher.
"