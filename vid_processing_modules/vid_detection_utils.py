# vid_detection_utils.py

import os
import gc
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog

import joblib
import librosa
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from feature_extraction import extract_sRQA_features_from_audio

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
FEATURES = [
    "peak_match",
    "peak_energy",
    "total_band_energy",
    "concentration",
    "RR",
    "DET",
    "L",
    "Lmax",
    "DIV",
    "ENTR",
    "LAM",
    "TT",
    "Vmax",
    "VENTR",
    "MRT",
    "RTE",
    "NMPRT",
    "TREND",
]
TARGET_SR = 16000


def select_base_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing videos")
    root.destroy()
    return folder


def select_airhorn_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select reference airhorn file",filetypes=[("Audio/Video files", "*.mp4 *.wav *.mp3 *.m4a *.avi *.mov *.mkv"),("All files", "*.*"),],)
    root.destroy()
    return file_path


def setup_paths(base_folder):
    return base_folder, os.path.join(base_folder, "horn_audios")


def verify_paths(video_folder, audio_folder):
    if not os.path.exists(video_folder):
        print(f"Missing video folder: {video_folder}")
        return False

    os.makedirs(audio_folder, exist_ok=True)
    return True


def find_video_files(video_folder):
    return [
        f for f in os.listdir(video_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]


def extract_audio_from_videos(video_files, video_folder, audio_folder):
    successful = 0
    failed = []

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        stem = os.path.splitext(video_file)[0]
        audio_path = os.path.join(audio_folder, f"{stem}.wav")

        if os.path.exists(audio_path):
            successful += 1
            continue

        try:
            video = VideoFileClip(video_path)

            if video.audio is None:
                failed.append(video_file)
                video.close()
                continue

            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()
            gc.collect()
            successful += 1

        except Exception:
            failed.append(video_file)

    return successful, failed


def get_airhorn_path(airhorn_file=None):
    airhorn_path = airhorn_file if airhorn_file else select_airhorn_file()

    if not airhorn_path or not os.path.exists(airhorn_path):
        return None

    return airhorn_path


def load_audio_from_video(path):
    video = VideoFileClip(path)

    if video.audio is None:
        video.close()
        return None, None

    temp_path = os.path.join(tempfile.gettempdir(), "temp_reference_audio.wav")
    video.audio.write_audiofile(temp_path, verbose=False, logger=None)
    video.close()

    audio, sr = librosa.load(temp_path, sr=TARGET_SR)
    os.remove(temp_path)

    return audio, sr


def load_airhorn_audio(airhorn_path):
    ext = os.path.splitext(airhorn_path)[1].lower()

    if ext in VIDEO_EXTENSIONS:
        return load_audio_from_video(airhorn_path)

    return librosa.load(airhorn_path, sr=TARGET_SR)


def calculate_harmonic_frequencies(freqs, top_indices, band_mask, harmonic_multipliers):
    band_freqs = freqs[band_mask]
    harmonic_indices = []

    for idx in top_indices:
        base_freq = band_freqs[idx]

        for h in harmonic_multipliers:
            target_freq = base_freq * h
            closest_idx = np.argmin(np.abs(band_freqs - target_freq))
            harmonic_indices.append(closest_idx)

    return np.unique(harmonic_indices)


def extract_window_features(window, band_mask, horn_band, harmonic_indices):
    window_fft = np.abs(np.fft.rfft(window))
    window_band = window_fft[band_mask]

    peak_match = np.dot(horn_band[harmonic_indices],window_band[harmonic_indices],)
    peak_energy = np.sum(window_band[harmonic_indices])
    total_band_energy = np.sum(window_band) + 1e-12
    concentration = peak_energy / total_band_energy
    raw_score = peak_match * concentration

    return {
        "peak_match": float(peak_match),
        "peak_energy": float(peak_energy),
        "total_band_energy": float(total_band_energy),
        "concentration": float(concentration),
        "raw_score": float(raw_score),
    }


def score_window(features, model=None):
    if model is None:
        return features["raw_score"], None

    feature_row = pd.DataFrame([{col: features[col] for col in FEATURES}])
    model_probability = float(model.predict_proba(feature_row)[0, 1])

    return model_probability, model_probability


def sliding_window_detection(video_audio,window_size,hop_size,band_mask,horn_band,harmonic_indices,model=None,):
    best_raw_score = -np.inf
    best_model_prob = -np.inf
    best_sample = 0
    best_features = None
    candidate_windows = []

    for start in range(0, len(video_audio) - window_size, hop_size):
        window = video_audio[start:start + window_size]

        features = extract_window_features(window,band_mask,horn_band,harmonic_indices,)
        raw_score = features["raw_score"]

        if model is None:
            if raw_score > best_raw_score:
                best_raw_score = raw_score
                best_sample = start
                best_features = features
        else:
            candidate_windows.append({
                "score": raw_score,
                "window": window.copy(),
                "start": start,
                "fft_features": {k: features[k] for k in features if k != "raw_score"},
            })

    if model is not None:
        candidate_windows = sorted(candidate_windows,key=lambda x: x["score"],reverse=True,)[:300]

        for candidate in candidate_windows:
            srqa_features = extract_sRQA_features_from_audio(candidate["window"],sr=16000,)

            window_srqa_features = {k: v for k, v in srqa_features.items()if k not in ["symbols", "rms_sequence", "recurrence_matrix"]}

            combined_features = {**candidate["fft_features"],**window_srqa_features,}

            model_prob = float(model.predict_proba(pd.DataFrame([combined_features]))[0, 1])

            if model_prob > best_model_prob:
                best_model_prob = model_prob
                best_raw_score = candidate["score"]
                best_sample = candidate["start"]
                best_features = combined_features

    if model is None:
        return best_raw_score, best_sample, None, best_features
    else:
        return best_raw_score, best_sample, best_model_prob, best_features

def detect_horn(video_audio, airhorn, sr, model=None):
    window_size = int(sr * 1.0)
    hop_size = int(sr * 0.05)

    airhorn = airhorn[:window_size]

    if len(airhorn) < window_size:
        airhorn = np.pad(airhorn, (0, window_size - len(airhorn)))

    horn_fft = np.abs(np.fft.rfft(airhorn))
    freqs = np.fft.rfftfreq(window_size, d=1 / sr)

    band_mask = (freqs >= 640) & (freqs <= 3400)
    horn_band = horn_fft[band_mask]

    if horn_band.size == 0 or np.max(horn_band) == 0:
        raise ValueError("Reference airhorn has no usable signal in the selected band.")

    threshold = 0.15 * np.max(horn_band)
    top_indices = np.where(horn_band >= threshold)[0]

    harmonic_indices = calculate_harmonic_frequencies(
        freqs=freqs,
        top_indices=top_indices,
        band_mask=band_mask,
        harmonic_multipliers=[1, 2, 3],
    )

    best_score, best_sample, model_probability, best_features = sliding_window_detection(video_audio=video_audio,window_size=window_size,hop_size=hop_size,band_mask=band_mask,horn_band=horn_band,harmonic_indices=harmonic_indices,model=model)

    horn_time = best_sample / sr

    return horn_time, best_score, model_probability, best_features


def crop_video(video_path, output_path, start, end):
    duration = end - start

    cmd = ["ffmpeg","-y","-ss",str(start),"-i",video_path,"-t",str(duration),"-c","copy",output_path,]

    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,)

    return os.path.exists(output_path), output_path


def process_single_video(video_file,video_folder,audio_folder,airhorn,sr_horn,output_folder,model=None,):
    stem = os.path.splitext(video_file)[0]
    video_path = os.path.join(video_folder, video_file)
    audio_path = os.path.join(audio_folder, f"{stem}.wav")
    output_path = os.path.join(output_folder, f"{stem}_cut.mp4")

    if not os.path.exists(audio_path):
        return None

    video_audio, sr = librosa.load(audio_path, sr=sr_horn)

    horn_time, score, model_probability, features = detect_horn(video_audio=video_audio,airhorn=airhorn,sr=sr,model=model,)
    video = VideoFileClip(video_path)
    start = max(horn_time - 10, 0)
    end = min(horn_time + 120, video.duration)
    video.close()

    success, final_path = crop_video(video_path, output_path, start, end)

    if not success:
        return None

    return {
        "video_file": video_file,
        "horn_time": horn_time,
        "clip_start": start,
        "clip_end": end,
        "clip_duration": end - start,
        "score": score,
        "model_probability": model_probability,
        "selection_method": "model" if model is not None else "raw_score",
        "peak_match": features["peak_match"],
        "peak_energy": features["peak_energy"],
        "total_band_energy": features["total_band_energy"],
        "concentration": features["concentration"],
        "output_path": final_path,
    }


def detect_horn_and_crop_videos(video_files,video_folder,audio_folder,base_folder,airhorn_file=None,model_path=None,):
    airhorn_path = get_airhorn_path(airhorn_file)

    if airhorn_path is None:
        return 0, []

    airhorn, sr_horn = load_airhorn_audio(airhorn_path)

    if airhorn is None:
        return 0, []

    model = joblib.load(model_path) if model_path else None

    output_folder = os.path.join(base_folder, "cropped_videos")
    os.makedirs(output_folder, exist_ok=True)

    results = []

    for video_file in video_files:
        result = process_single_video(
            video_file=video_file,
            video_folder=video_folder,
            audio_folder=audio_folder,
            airhorn=airhorn,
            sr_horn=sr_horn,
            output_folder=output_folder,
            model=model,
        )

        if result is not None:
            results.append(result)

    csv_path = None

    if results:
        csv_path = os.path.join(base_folder, "horn_detection_results.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)

    return len(results), results, output_folder, csv_path, airhorn_path


def generate_summary_report(base_folder,video_files,successful_extractions,failed_extractions,successful_crops=None,processing_results=None,cropped_video_folder=None,results_csv_path=None,airhorn_path=None,audio_folder=None,):
    print(f"Videos found: {len(video_files)}")
    print(f"Audio extracted: {successful_extractions}")
    print(f"Audio failed: {len(failed_extractions)}")

    if successful_crops is not None:
        print(f"Crops created: {successful_crops}")

    if results_csv_path:
        print(f"Results CSV: {results_csv_path}")