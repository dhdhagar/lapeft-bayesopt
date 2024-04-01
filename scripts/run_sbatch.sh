#!/bin/bash -e

echo CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}
eval "$(conda shell.bash hook)"
conda deactivate
conda activate lapeft
export PYTHONPATH=$(pwd):$PYTHONPATH;
export TRANSFORMERS_CACHE=/iesl/canvas/dagarwal/hf_cache
python "$@"
