import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
from src.inference.predict import predict_from_folder
from src.config import CLASSES
import numpy as np

st.set_page_config(page_title="Baby Sound Detection", layout="wide")

st.title("🍼 Baby Sound Detection Dashboard")
st.write("Automatically fetching and analyzing all baby sounds from the **sampletest/** folder (split into 7-second clips).")

folder_path = "sampletest"
checkpoint_path = "checkpoints/best_epoch_30.pth"

# Run prediction automatically
with st.spinner("Analyzing baby sounds... Please wait..."):
    results = predict_from_folder(folder_path, checkpoint_path)

if results:
    for res in results:
        st.markdown(f"### 🎧 {res['clip']}")
        audio_file_path = os.path.join(folder_path, res['file'])
        if os.path.exists(audio_file_path):
            st.audio(audio_file_path, format="audio/wav")

        st.success(f"**Predicted Sound:** {res['label'].upper()}")
        st.bar_chart({cls: float(np.round(p, 3)) for cls, p in zip(CLASSES, res['probs'])})
        st.info(f"💡 **Recommendation:** {res['recommendation']}")
        st.markdown("---")
else:
    st.warning("⚠️ No audio files found in 'sampletest' folder.")
