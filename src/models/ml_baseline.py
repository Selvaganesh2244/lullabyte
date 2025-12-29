# src/models/ml_baseline.py
# This contains helpers for feature extraction for classical ML models (XGBoost/RandomForest)
import numpy as np
import librosa

def extract_mfcc_features(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # take mean and std across time
    feats = []
    feats.extend(mfcc.mean(axis=1).tolist())
    feats.extend(mfcc.std(axis=1).tolist())
    return np.array(feats)
