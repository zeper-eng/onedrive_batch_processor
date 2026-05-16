import os
import sys
import numpy as np
import pandas as pd
import librosa

from feature_extraction import (load_audio_chunk,prepare_horn_template,extract_detector_features,extract_sRQA_features_from_audio,)
from vid_detection_utils import select_base_folder, find_video_files

# Replace with your actual failed cut filenames, or load from a file.
# These are the videos where the detector produced incorrect crops (label=0).
FAILED_CUTS = ["video_003.mp4", "video_007.mp4", "video_012.mp4"]

base_folder = sys.argv[1] if len(sys.argv) > 1 else select_base_folder()
airhorn_file = sys.argv[2] if len(sys.argv) > 2 else None
csv_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(base_folder,"horn_training_features.csv")

TARGET_SR = 16000

airhorn_audio, sr = librosa.load(airhorn_file, sr=TARGET_SR, mono=True)
horn_template = prepare_horn_template(airhorn_audio, sr)
video_files = find_video_files(base_folder)

WINDOWS = [(9.5, 10.5), (10.5, 11.5)]

rows = []

for video_file in video_files:
    video_path = os.path.join(base_folder, video_file)
    label = 0 if video_file in FAILED_CUTS else 1

    for start_sec, end_sec in WINDOWS:
        audio_chunk, sr = load_audio_chunk(video_path,start_sec=start_sec,end_sec=end_sec,sr=TARGET_SR,)

        # sRQA features: instead of raw waveforms, we use RMS energy sequences
        # as input to symbolic recurrence analysis. RMS smooths out the rapid
        # oscillation of the waveform and preserves the energy envelope shape,
        # which is more stable and informative for detecting the horn onset.
        window_sRQA_features = extract_sRQA_features_from_audio(audio_chunk, sr)
        window_sRQA_features = {k: v for k, v in window_sRQA_features.items()if k not in ["symbols", "rms_sequence", "recurrence_matrix"]}
        features = extract_detector_features(audio_chunk=audio_chunk,horn_template=horn_template,)
        rows.append({"video_file": video_file,"window_start": start_sec,"window_end": end_sec,**features,**window_sRQA_features,"label": label,})

pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
print(f"Saved: {csv_path}")

df = pd.read_csv(csv_path)
df = df.drop_duplicates()
df.to_csv(csv_path, index=False)
print(f"Cleared any potential duplicates inside of: {csv_path}")