"""Make Heretic pass trust_remote_code=True for models with custom code."""
import os
import sys

HERETIC_DIR = os.path.join(
    sys.prefix, "Lib", "site-packages", "heretic"
)


def patch_file(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    for old, new in replacements:
        if old not in src:
            print(f"  [skip] pattern not found in {path}: {old[:60]!r}")
            continue
        src = src.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched {path}")


def main():
    model_py = os.path.join(HERETIC_DIR, "model.py")
    main_py = os.path.join(HERETIC_DIR, "main.py")

    patch_file(model_py, [
        (
            "    configs = PretrainedConfig.get_config_dict(model)",
            "    configs = PretrainedConfig.get_config_dict(model, trust_remote_code=True)",
        ),
        (
            """                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True
                    if settings.model in self.trusted_models
                    else None,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )""",
            """                self.model = get_model_class(settings.model).from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=True,
                    **self.revision_kwargs,
                    **extra_kwargs,
                )""",
        ),
        (
            """            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True
                if self.settings.model in self.trusted_models
                else None,
                **self.revision_kwargs,
            )""",
            """            base_model = get_model_class(self.settings.model).from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=True,
                **self.revision_kwargs,
            )""",
        ),
        (
            """        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True
            if self.settings.model in self.trusted_models
            else None,
            **self.revision_kwargs,
            **extra_kwargs,
        )""",
            """        self.model = get_model_class(self.settings.model).from_pretrained(
            self.settings.model,
            dtype=self.dtype,
            device_map=self.settings.device_map,
            max_memory=self.max_memory,
            trust_remote_code=True,
            **self.revision_kwargs,
            **extra_kwargs,
        )""",
        ),
    ])

    # Tokenizer / processor
    with open(model_py, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace(
        """            AutoTokenizer.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            ),""",
        """            AutoTokenizer.from_pretrained(
                settings.model,
                trust_remote_code=True,
                **self.revision_kwargs,
            ),""",
    )
    src = src.replace(
        """            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                **self.revision_kwargs,
            )""",
        """            self.processor = AutoProcessor.from_pretrained(
                settings.model,
                trust_remote_code=True,
                **self.revision_kwargs,
            )""",
    )
    with open(model_py, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"Patched {model_py} (tokenizer/processor)")

    patch_file(main_py, [
        (
            """                    trust_remote_code=True
                    if settings.model in model.trusted_models
                    else None,
                    **model.revision_kwargs,""",
            """                    trust_remote_code=True,
                    **model.revision_kwargs,""",
        ),
    ])


if __name__ == "__main__":
    main()
"