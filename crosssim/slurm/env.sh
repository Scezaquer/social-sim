# Shared environment for the cross-simulator experiments (sourced by every script).
export ENVS="${ENVS:-$SCRATCH/crosssim_envs}"          # python venvs (vllm, tools, oasis, concordia, silisocs)
export CROSSSIM_OUT="${CROSSSIM_OUT:-$SCRATCH/crosssim_out}"   # full design + run outputs
export CROSSSIM_SMOKE="${CROSSSIM_SMOKE:-$SCRATCH/crosssim_smoke}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/HF-cache}"  # same cache as the paper's runs
export HF_HOME="${HF_HOME:-$SCRATCH/HF-home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/uv-cache}"
export UV_LINK_MODE=copy   # network FS: avoid hardlink surprises
export VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
export PATH="$HOME/.local/bin:$PATH"
export TRANSFORMERS_VERBOSITY=error   # silences "PyTorch was not found" in the tools env
