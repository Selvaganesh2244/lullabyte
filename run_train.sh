#!/bin/bash

# Training script for Lullabyte model

# Activate virtual environment
source venv/bin/activate

# Run training
python -m src.training.train \
    --model efficientnet_lstm \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.001 \
    --early_stopping 10