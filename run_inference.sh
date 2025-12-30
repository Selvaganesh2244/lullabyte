#!/bin/bash

# Inference script for Lullabyte model

# Activate virtual environment
source venv/bin/activate

# Check if audio file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path/to/audio.wav>"
    exit 1
fi

# Run inference
python -m src.inference.predict \
    --model_path experiments/checkpoints/best_model.pth \
    --audio_file "$1"