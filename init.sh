#!/bin/bash

conda activate lapeft
export TRANSFORMERS_CACHE=/iesl/canvas/dagarwal/hf_cache
export PYTHONPATH=$(pwd):$PYTHONPATH
