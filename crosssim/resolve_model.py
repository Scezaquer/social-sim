"""Find a local copy of a Hugging Face model (config + tokenizer + weights) and print its path.

Looks in every usual cache location (HF_HUB_CACHE, HF_HOME/hub, $SCRATCH/HF-cache,
~/.cache/huggingface/hub, ...) and in plain directories such as $SCRATCH/<org>/<name>.
With --download, falls back to snapshot_download into $HF_HUB_CACHE (needs internet).

    python crosssim/resolve_model.py marcelbinz/Llama-3.1-Minitaur-8B [--download]
"""
import glob
import os
import sys
from pathlib import Path


def complete(d: Path) -> bool:
    has_cfg = (d / "config.json").is_file()
    has_tok = any((d / f).exists() for f in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
    has_w = bool(list(d.glob("*.safetensors")) or list(d.glob("*.bin")))
    return has_cfg and has_tok and has_w


def candidates(repo: str):
    org, _, name = repo.partition("/")
    env = os.environ
    scratch, home = env.get("SCRATCH", ""), str(Path.home())
    hub_dirs = [env.get("HF_HUB_CACHE"), env.get("HUGGINGFACE_HUB_CACHE"),
                env.get("HF_HOME") and os.path.join(env["HF_HOME"], "hub"),
                scratch and f"{scratch}/HF-cache", scratch and f"{scratch}/HF-cache/hub",
                scratch and f"{scratch}/huggingface/hub", scratch and f"{scratch}/.cache/huggingface/hub",
                scratch and f"{scratch}/hf_cache", scratch and f"{scratch}/hf_cache/hub",
                f"{home}/.cache/huggingface/hub", f"{home}/scratch/HF-cache"]
    key = "models--" + repo.replace("/", "--")
    for h in hub_dirs:
        if h:
            for snap in sorted(glob.glob(f"{h}/{key}/snapshots/*"), key=os.path.getmtime, reverse=True):
                yield Path(snap)
    for base in (scratch, home, f"{home}/scratch"):
        if base:
            yield Path(base) / org / name
            yield Path(base) / name
    if scratch:  # last resort: any cache dir for this repo up to 3 levels under $SCRATCH
        for snap in glob.glob(f"{scratch}/*/{key}/snapshots/*") + glob.glob(f"{scratch}/*/*/{key}/snapshots/*"):
            yield Path(snap)


def main():
    repo = sys.argv[1]
    if os.path.isdir(repo) and complete(Path(repo)):
        print(os.path.abspath(repo)); return
    seen = []
    for d in candidates(repo):
        seen.append(str(d))
        if d.is_dir() and complete(d):
            print(d); return
    if "--download" in sys.argv:
        from huggingface_hub import snapshot_download
        cache = os.environ.get("HF_HUB_CACHE") or None
        print(f"[resolve_model] {repo} not found locally; downloading to {cache}", file=sys.stderr)
        p = snapshot_download(repo, cache_dir=cache, allow_patterns=[
            "*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.jinja"])
        if complete(Path(p)):
            print(p); return
    print(f"[resolve_model] could not find {repo}. Looked in:\n  " + "\n  ".join(dict.fromkeys(seen)), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
