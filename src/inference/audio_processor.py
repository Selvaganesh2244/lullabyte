"""
Audio processing utilities for inference.
"""

import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from src.config import SR

def split_audio(file_path: str, segment_duration: float = 7.0) -> Tuple[List[Tuple[np.ndarray, str]], int]:
    """
    Split audio file into fixed-duration segments.
    
    Args:
        file_path: Path to audio file
        segment_duration: Duration of each segment in seconds
        
    Returns:
        Tuple containing:
            - List of (audio_segment, segment_name) tuples
            - Sample rate
    """
    try:
        y, sr = librosa.load(file_path, sr=SR)
        segment_samples = int(segment_duration * sr)
        clips = []
        
        total_segments = int(np.ceil(len(y) / segment_samples))
        for i in range(total_segments):
            start = int(i * segment_samples)
            end = int(min(len(y), (i + 1) * segment_samples))
            clip = y[start:end]
            
            # Create segment name
            stem = Path(file_path).stem
            segment_name = f"{stem}_part{i+1}.wav"
            
            clips.append((clip, segment_name))
            
        return clips, sr
        
    except Exception as e:
        raise Exception(f"Error processing audio file {file_path}: {str(e)}")

def find_audio_files(folder_path: str) -> List[Path]:
    """
    Find all audio files in a folder.
    
    Args:
        folder_path: Path to folder to search
        
    Returns:
        List of paths to audio files
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} not found")
        
    audio_files = list(folder.glob("*.wav")) + list(folder.glob("*.mp3"))
    return audio_files