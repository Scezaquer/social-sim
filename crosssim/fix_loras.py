"""Inspect the BluePrint LoRA adapters and write vLLM-ready copies.

vLLM 0.11 crashes (tensor-size mismatch in the LoRA logits processor, extra-vocab size 256) when an
adapter ships extra-vocabulary embeddings (new_embeddings.safetensors / added tokens) and a request
needs logits beyond plain sampling (logprobs, allowed_token_ids, structured tool-call output).
This writes $SCRATCH/crosssim_loras/<family>/lora<i>/ containing ONLY adapter_config.json and the
adapter weights (tokenizer files, added_tokens.json and new_embeddings.* are left out). The adapter
weights themselves are unchanged unless --strip-vocab-modules is given, which also drops LoRA/
full-weight tensors on embed_tokens / lm_head (use only if the plain copy still crashes).

    python crosssim/fix_loras.py <family> <lora_path_template_with_{i}> [--strip-vocab-modules]
"""
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from safetensors import safe_open

fam, template = sys.argv[1], sys.argv[2]
strip = "--strip-vocab-modules" in sys.argv
out_root = Path(os.environ["SCRATCH"]) / "crosssim_loras" / fam
VOCAB_KEYS = ("embed_tokens", "lm_head", "wte", "embed_out")

for i in range(25):
    src = Path(template.format(i=i))
    dst = out_root / f"lora{i}"
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "adapter_config.json").read_text())
    files = sorted(p.name for p in src.iterdir())
    wfile = src / "adapter_model.safetensors"
    with safe_open(str(wfile), "np" if not strip else "pt") as f:
        keys = list(f.keys())
        vocab_keys = [k for k in keys if any(v in k for v in VOCAB_KEYS)]
        if i == 0:
            print(f"[fix_loras] {fam} adapter 0: files={files}")
            print(f"[fix_loras]   r={cfg.get('r')} target_modules={cfg.get('target_modules')} "
                  f"modules_to_save={cfg.get('modules_to_save')} n_tensors={len(keys)}")
            print(f"[fix_loras]   embedding/lm_head tensors: {vocab_keys[:6]}{' ...' if len(vocab_keys) > 6 else ''}")
        if strip and vocab_keys:
            tensors = {k: f.get_tensor(k) for k in keys if k not in vocab_keys}
            from safetensors.torch import save_file as _save
            _save(tensors, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})
            tm = cfg.get("target_modules")
            if isinstance(tm, list):
                cfg["target_modules"] = [m for m in tm if m not in VOCAB_KEYS]
            cfg["modules_to_save"] = None
        else:
            if (dst / "adapter_model.safetensors").exists() or (dst / "adapter_model.safetensors").is_symlink():
                (dst / "adapter_model.safetensors").unlink()
            os.symlink(wfile, dst / "adapter_model.safetensors")
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    dropped = [n for n in files if n not in ("adapter_config.json", "adapter_model.safetensors")]
    if i == 0:
        print(f"[fix_loras]   left out of the vLLM copy: {dropped}")
print(f"[fix_loras] wrote 25 adapters -> {out_root} (strip_vocab_modules={strip})")
