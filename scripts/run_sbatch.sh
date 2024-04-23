#!/bin/bash -e

echo CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}
eval "$(conda shell.bash hook)"
conda deactivate
conda activate lapeft
export PYTHONPATH=$(pwd):$PYTHONPATH;
export TRANSFORMERS_CACHE=/project/pi_mccallum_umass_edu/dagarwal_umass_edu/huggingface_cache
# Unity: /project/pi_mccallum_umass_edu/dagarwal_umass_edu/huggingface_cache
# Blake: /iesl/canvas/dagarwal/hf_cache
python "$@"
