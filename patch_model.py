"""Patch Nanbeige's custom modeling code for transformers 5.x."""
import argparse
import os


def patch(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new in replacements:
        if old not in src:
            print(f"  [skip] pattern not found: {old[:60]!r}")
            continue
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    modeling = os.path.join(args.model_dir, "modeling_nanbeige.py")
    if not os.path.isfile(modeling):
        raise SystemExit(f"Not found: {modeling}")

    patch(modeling, [
        # GenerationMixin import
        (
            "from transformers.modeling_utils import PreTrainedModel",
            "from transformers.modeling_utils import PreTrainedModel\nfrom transformers.generation import GenerationMixin",
        ),
        # NanbeigeForCausalLM inherits GenerationMixin
        (
            "class NanbeigeForCausalLM(NanbeigePreTrainedModel):",
            "class NanbeigeForCausalLM(NanbeigePreTrainedModel, GenerationMixin):",
        ),
        # Guard get_max_length on empty cache
        (
            """            max_cache_length = (
                torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                if past_key_values.get_max_length() is not None
                else None
            )""",
            """            try:
                _max_length = past_key_values.get_max_length()
            except (ValueError, StopIteration, RuntimeError):
                _max_length = None
            if _max_length is None or _max_length == -1:
                max_cache_length = None
            else:
                try:
                    max_cache_length = torch.tensor(_max_length, device=input_ids.device)
                except Exception:
                    max_cache_length = None""",
        ),
        # Slice position_ids carried over from previous steps
        (
            """        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:""",
            """        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None and past_key_values is not None:
            if position_ids.shape[-1] != input_ids.shape[-1]:
                position_ids = position_ids[..., -input_ids.shape[-1]:].clone()
        if attention_mask is not None and position_ids is None:""",
        ),
    ])


if __name__ == "__main__":
    main()
"