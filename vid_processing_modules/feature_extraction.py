import librosa
import numpy as np

def load_audio_chunk(video_path, start_sec, end_sec, sr=None):
    audio, sample_rate = librosa.load(
        video_path,
        sr=sr,
        mono=True,
        offset=start_sec,
        duration=end_sec - start_sec
    )

    return audio, sample_rate


def calculate_harmonic_frequencies(freqs, top_indices, band_mask, harmonic_multipliers):
    harmonic_indices = []

    band_freqs = freqs[band_mask]

    for idx in top_indices:
        base_freq = band_freqs[idx]

        for h in harmonic_multipliers:
            target_freq = base_freq * h
            closest_idx = np.argmin(np.abs(band_freqs - target_freq))
            harmonic_indices.append(closest_idx)

    return np.unique(harmonic_indices)


def prepare_horn_template(airhorn_audio, sr):
    window_size = int(sr * 1.0)

    airhorn_audio = airhorn_audio[:window_size]

    horn_fft = np.abs(np.fft.rfft(airhorn_audio))
    freqs = np.fft.rfftfreq(window_size, d=1 / sr)

    band_mask = (freqs >= 640) & (freqs <= 3400)
    horn_band = horn_fft[band_mask]

    threshold = 0.15 * np.max(horn_band)
    top_indices = np.where(horn_band >= threshold)[0]

    harmonic_indices = calculate_harmonic_frequencies(
        freqs=freqs,
        top_indices=top_indices,
        band_mask=band_mask,
        harmonic_multipliers=[1, 2, 3]
    )

    return {
        "window_size": window_size,
        "freqs": freqs,
        "band_mask": band_mask,
        "horn_band": horn_band,
        "harmonic_indices": harmonic_indices,
    }


def extract_detector_features(audio_chunk, horn_template):
    eps = 1e-12

    window_size = horn_template["window_size"]
    band_mask = horn_template["band_mask"]
    horn_band = horn_template["horn_band"]
    harmonic_indices = horn_template["harmonic_indices"]

    if len(audio_chunk) < window_size:
        return {
            "peak_match": 0,
            "peak_energy": 0,
            "total_band_energy": 0,
            "concentration": 0,
            "peak_energy_ratio": 0,
        }

    window = audio_chunk[:window_size]

    window_fft = np.abs(np.fft.rfft(window))
    window_band = window_fft[band_mask]

    peak_match = np.dot(
        horn_band[harmonic_indices],
        window_band[harmonic_indices]
    )

    peak_energy = np.sum(window_band[harmonic_indices])
    total_band_energy = np.sum(window_band) + eps
    concentration = peak_energy / total_band_energy
    raw_score = peak_match * concentration

    return {
        "peak_match": float(peak_match),
        "peak_energy": float(peak_energy),
        "total_band_energy": float(total_band_energy),
        "concentration": float(concentration),
        "peak_energy_ratio": float(peak_energy / total_band_energy),
    }