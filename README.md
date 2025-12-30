# Lullabyte: Infant Cry Classification

A deep learning system for classifying infant cries using audio processing and neural networks.

## Project Structure

```
Lullabyte/
│── data/
│   ├── raw/                 # Original infant audio files
│   ├── processed/           # Preprocessed audio files
│   ├── spectrograms/        # Generated spectrograms
│   └── labels.csv           # Metadata and labels
│
│── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
│── src/                     # Source code
│   ├── data_preprocessing/  # Data loading and processing
│   ├── models/             # Model architectures
│   ├── training/           # Training utilities
│   ├── inference/          # Inference scripts
│   └── visualization/      # Visualization tools
│
│── experiments/            # Experiment tracking
    ├── logs/              # TensorBoard logs
    ├── checkpoints/       # Model checkpoints
    └── results/           # Evaluation results
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place raw audio files in `data/raw/`
2. Run preprocessing:
```bash
python src/data_preprocessing/audio_loader.py
```

### Training
1. Configure hyperparameters in `src/config.py`
2. Start training:
```bash
./run_train.sh
```

### Inference
1. Run predictions on new audio:
```bash
./run_inference.sh path/to/audio.wav
```

## Models

1. EfficientNet + LSTM Hybrid
- CNN feature extraction
- LSTM temporal modeling
- State-of-the-art performance

2. CNN Baseline
- Simple convolutional architecture
- Good baseline performance

3. Classical ML Baseline
- XGBoost/Random Forest on MFCCs
- Quick experimentation

## License

MIT License