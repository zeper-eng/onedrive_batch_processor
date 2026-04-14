import os
import sys
import gc
import numpy as np
import pandas as pd
import librosa
import scipy.signal
from moviepy.editor import VideoFileClip

def find_videos(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    if video.audio is None:
        video.close()
        return False
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    gc.collect()
    return True

def detect_and_crop(video_path, audio_path, ref_audio, sr_ref, out_path):
    audio, sr = librosa.load(audio_path, sr=sr_ref)
    corr = scipy.signal.correlate(audio, ref_audio, mode="valid")
    horn_start = np.argmax(corr) / sr

    video = VideoFileClip(video_path)
    start = max(horn_start - 10, 0)
    end = min(horn_start + 120, video.duration)

    clip = video.subclip(start, end)
    clip.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    clip.close()
    video.close()
    gc.collect()

    return {"video": os.path.basename(video_path), "horn_time": horn_start, "start": start, "end": end}

def main():
    base_folder = sys.argv[1]
    ref_path = sys.argv[2]

    audio_folder = os.path.join(base_folder, "horn_audios")
    out_folder = os.path.join(base_folder, "cropped_videos")
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    ref_audio, sr_ref = librosa.load(ref_path, sr=None)
    videos = find_videos(base_folder)

    results = []

    for name in videos:
        video_path = os.path.join(base_folder, name)
        stem = os.path.splitext(name)[0]
        audio_path = os.path.join(audio_folder, f"{stem}.wav")
        out_path = os.path.join(out_folder, f"{stem}_cut.mp4")

        if not os.path.exists(audio_path):
            ok = extract_audio(video_path, audio_path)
            if not ok:
                continue

        if os.path.exists(out_path):
            continue

        try:
            row = detect_and_crop(video_path, audio_path, ref_audio, sr_ref, out_path)
            results.append(row)
        except Exception as e:
            print(f"Failed: {name} -> {e}")

    if results:
        pd.DataFrame(results).to_csv(os.path.join(base_folder, "horn_detection_results.csv"), index=False)

if __name__ == "__main__":
    main()