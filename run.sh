#!/bin/bash

set -e

echo "Starting full pipeline..."

echo "Starting BINN1 pipeline..."
bash binn1_pixelreg/repro.sh

echo "Completed all experiments successfully"