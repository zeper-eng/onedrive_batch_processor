# horn_detection_pipeline.py

import os
import sys
import gc
import subprocess

import librosa
import numpy as np
import pandas as pd
import joblib
from moviepy.editor import VideoFileClip


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def find_videos(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]


def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)

    if video.audio is None:
        video.close()
        return False

    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    gc.collect()

    return True


def load_reference_audio(ref_path):
    ext = os.path.splitext(ref_path)[1].lower()

    if ext in VIDEO_EXTENSIONS:
        video = VideoFileClip(ref_path)

        if video.audio is None:
            video.close()
            raise ValueError(f"Reference video has no audio: {ref_path}")

        temp_path = os.path.join(
            os.path.dirname(ref_path),
            "_temp_reference_audio.wav"
        )

        video.audio.write_audiofile(temp_path, verbose=False, logger=None)
        video.close()

        audio, sr = librosa.load(temp_path, sr=None)
        os.remove(temp_path)

        return audio, sr

    return librosa.load(ref_path, sr=None)


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

    peak_match = np.dot(
        horn_band[harmonic_indices],
        window_band[harmonic_indices]
    )

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

    model_features = pd.DataFrame([{
        "peak_match": features["peak_match"],
        "peak_energy": features["peak_energy"],
        "total_band_energy": features["total_band_energy"],
        "concentration": features["concentration"],
    }])

    model_probability = float(model.predict_proba(model_features)[0, 1])
    return model_probability, model_probability


def sliding_window_detection(audio, window_size, hop_size, band_mask, horn_band, harmonic_indices, model=None):
    best_score = -np.inf
    best_sample = 0
    best_features = None
    best_model_probability = None

    for start in range(0, len(audio) - window_size, hop_size):
        window = audio[start:start + window_size]

        features = extract_window_features(
            window,
            band_mask,
            horn_band,
            harmonic_indices
        )

        score, model_probability = score_window(features, model=model)

        if score > best_score:
            best_score = score
            best_sample = start
            best_features = features
            best_model_probability = model_probability

    return best_sample, best_score, best_model_probability, best_features


def detect_horn(audio, reference_audio, sr, model=None):
    window_size = int(sr * 1.0)
    hop_size = int(sr * 0.05)

    reference_audio = reference_audio[:window_size]

    if len(reference_audio) < window_size:
        reference_audio = np.pad(
            reference_audio,
            (0, window_size - len(reference_audio))
        )

    horn_fft = np.abs(np.fft.rfft(reference_audio))
    freqs = np.fft.rfftfreq(window_size, d=1 / sr)

    band_mask = (freqs >= 640) & (freqs <= 3400)
    horn_band = horn_fft[band_mask]

    if horn_band.size == 0 or np.max(horn_band) == 0:
        raise ValueError("Reference audio has no usable signal in the selected frequency band.")

    threshold = 0.15 * np.max(horn_band)
    top_indices = np.where(horn_band >= threshold)[0]

    harmonic_indices = calculate_harmonic_frequencies(
        freqs=freqs,
        top_indices=top_indices,
        band_mask=band_mask,
        harmonic_multipliers=[1, 2, 3]
    )

    best_sample, best_score, model_probability, best_features = sliding_window_detection(
        audio=audio,
        window_size=window_size,
        hop_size=hop_size,
        band_mask=band_mask,
        horn_band=horn_band,
        harmonic_indices=harmonic_indices,
        model=model
    )

    horn_time = best_sample / sr

    return horn_time, best_score, model_probability, best_features


def crop_video(video_path, out_path, start, end):
    duration = end - start

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        out_path
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return os.path.exists(out_path)


def detect_and_crop(video_path, audio_path, reference_audio, sr_ref, out_path, model=None):
    audio, sr = librosa.load(audio_path, sr=sr_ref)

    horn_time, score, model_probability, features = detect_horn(
        audio,
        reference_audio,
        sr,
        model=model
    )

    video = VideoFileClip(video_path)
    start = max(horn_time - 10, 0)
    end = min(horn_time + 120, video.duration)
    video.close()

    crop_video(video_path, out_path, start, end)
    gc.collect()

    return {
        "video": os.path.basename(video_path),
        "horn_time": horn_time,
        "start": start,
        "end": end,
        "score": score,
        "model_probability": model_probability,
        "selection_method": "model" if model is not None else "raw_score",
        "peak_match": features["peak_match"],
        "peak_energy": features["peak_energy"],
        "total_band_energy": features["total_band_energy"],
        "concentration": features["concentration"],
        "raw_score": features["raw_score"],
        "output_path": out_path,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python horn_detection_pipeline.py <base_folder> <reference_audio_or_video> [model_path]")
        sys.exit(1)

    base_folder = sys.argv[1]
    reference_path = sys.argv[2]
    model_path = sys.argv[3] if len(sys.argv) > 3 else None

    audio_folder = os.path.join(base_folder, "horn_audios")
    out_folder = os.path.join(base_folder, "cropped_videos")

    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    reference_audio, sr_ref = load_reference_audio(reference_path)
    model = joblib.load(model_path) if model_path else None

    videos = find_videos(base_folder)
    results = []

    for name in videos:
        video_path = os.path.join(base_folder, name)
        stem = os.path.splitext(name)[0]

        audio_path = os.path.join(audio_folder, f"{stem}.wav")
        out_path = os.path.join(out_folder, f"{stem}_cut.mp4")

        if not os.path.exists(audio_path):
            if not extract_audio(video_path, audio_path):
                continue

        if os.path.exists(out_path):
            continue

        try:
            row = detect_and_crop(
                video_path,
                audio_path,
                reference_audio,
                sr_ref,
                out_path,
                model=model
            )
            results.append(row)
        except Exception as e:
            print(f"Failed {name}: {e}")

    if results:
        results_path = os.path.join(base_folder, "horn_detection_results.csv")
        pd.DataFrame(results).to_csv(results_path, index=False)


if __name__ == "__main__":
    main()