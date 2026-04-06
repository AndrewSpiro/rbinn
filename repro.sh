#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

conda activate pixelreg

python repro.py