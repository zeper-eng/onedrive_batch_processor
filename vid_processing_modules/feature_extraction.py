import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

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

def save_recurrence_plot_colored(recurrence_matrix, symbols, output_name):
    out_dir = os.getcwd()

    symbols = np.asarray(symbols)

    # Where recurrence_matrix == 1, store the symbol value for that row.
    # Where recurrence_matrix == 0, store NaN so it plots as white.
    colored = np.where(recurrence_matrix == 1, symbols[:, None], np.nan)

    # For symbols 0,1,2,3,4; similar idea to their colored symbolic plot
    cmap = ListedColormap(["red", "yellow", "green", "cyan", "blue", "purple"])
    cmap.set_bad("white")

    plt.figure(figsize=(5, 5))
    plt.imshow(
        colored,
        cmap=cmap,
        origin="lower",
        interpolation="none"
    )
    plt.title(output_name)
    plt.xlabel("Time index")
    plt.ylabel("Time index")
    plt.colorbar(label="Symbol bin")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{output_name}_recurrence.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def extract_sRQA_features(R, lmin=2):

    R = np.asarray(R)
    N = R.shape[0]

    # Shannon entropy from empirical counts
    def shannon_entropy(values):
        if len(values) == 0:
            return 0.0

        _, counts = np.unique(values, return_counts=True)

        probs = counts / counts.sum()

        return -np.sum(probs * np.log(probs + 1e-12))

    # Diagonal line lengths
    diag_lengths = []

    for offset in range(-N + 1, N):
        diag = np.diagonal(R, offset=offset)

        run = 0

        for val in diag:
            if val == 1:
                run += 1
            else:
                if run >= lmin:
                    diag_lengths.append(run)
                run = 0

        if run >= lmin:
            diag_lengths.append(run)

    # Vertical line lengths
    vert_lengths = []

    for col in range(N):
        run = 0

        for row in range(N):
            if R[row, col] == 1:
                run += 1
            else:
                if run >= lmin:
                    vert_lengths.append(run)
                run = 0

        if run >= lmin:
            vert_lengths.append(run)

    # Recurrence counts
    total_points = N * N
    recurrent_points = np.sum(R)

    RR = recurrent_points / total_points

    # Diagonal measures
    diag_points = np.sum(diag_lengths)

    DET = diag_points / recurrent_points if recurrent_points > 0 else 0

    L = np.mean(diag_lengths) if diag_lengths else 0

    Lmax = np.max(diag_lengths) if diag_lengths else 0

    DIV = 1 / Lmax if Lmax > 0 else 0

    ENTR = shannon_entropy(diag_lengths)

    # Vertical measures
    vert_points = np.sum(vert_lengths)

    LAM = vert_points / recurrent_points if recurrent_points > 0 else 0

    TT = np.mean(vert_lengths) if vert_lengths else 0

    Vmax = np.max(vert_lengths) if vert_lengths else 0

    VENTR = shannon_entropy(vert_lengths)

    # Recurrence times
    recurrence_times = []

    for row in range(N):
        recurrence_indices = np.where(R[row] == 1)[0]

        if len(recurrence_indices) >= 2:
            recurrence_times.extend(
                np.diff(recurrence_indices)
            )

    recurrence_times = np.asarray(recurrence_times)

    MRT = np.mean(recurrence_times) if len(recurrence_times) > 0 else 0

    RTE = shannon_entropy(recurrence_times)

    NMPRT = len(recurrence_times)

    # TREND
    diagonal_rr = []

    for offset in range(N):
        diag = np.diagonal(R, offset=offset)

        if len(diag) > 0:
            diagonal_rr.append(np.mean(diag))

    diagonal_rr = np.asarray(diagonal_rr)

    if len(diagonal_rr) > 1:
        x = np.arange(len(diagonal_rr))
        TREND = np.polyfit(x, diagonal_rr, 1)[0]
    else:
        TREND = 0

    return {
        "RR": float(RR),

        "DET": float(DET),
        "L": float(L),
        "Lmax": float(Lmax),
        "DIV": float(DIV),
        "ENTR": float(ENTR),

        "LAM": float(LAM),
        "TT": float(TT),
        "Vmax": float(Vmax),
        "VENTR": float(VENTR),

        "MRT": float(MRT),
        "RTE": float(RTE),
        "NMPRT": int(NMPRT),

        "TREND": float(TREND),
    }

def extract_sRQA_features_from_audio(audio_chunk, sr, n_bins=5, m=1, d=1, lmin=2, frame_ms=20, hop_ms=10):

    rms_sequence = extract_rms_sequence(audio_chunk=audio_chunk, sr=sr, frame_ms=frame_ms, hop_ms=hop_ms)

    symbols = symbolize_rms_quantiles(rms_sequence, n_bins=n_bins)

    recurrence_matrix = create_recurrence_matrix(symbols, m=m, d=d).copy()

    np.fill_diagonal(recurrence_matrix, 0)

    features = extract_sRQA_features(recurrence_matrix, lmin=lmin)

    return {
        **features,
        "symbols": symbols,
        "rms_sequence": rms_sequence,
        "recurrence_matrix": recurrence_matrix,
    }

def prepare_horn_template(airhorn_audio, sr):
    window_size = int(sr * 1.0)

    airhorn_audio = airhorn_audio[:window_size]

    horn_fft = np.abs(np.fft.rfft(airhorn_audio))
    freqs = np.fft.rfftfreq(window_size, d=1 / sr)

    band_mask = (freqs >= 640) & (freqs <= 3400)#again this band is manually calculated from looking at spectrograms
    horn_band = horn_fft[band_mask]

    threshold = 0.15 * np.max(horn_band)
    top_indices = np.where(horn_band >= threshold)[0] #find strongest horn an dkeep everything atleast 15% as strong; this is to avoid picking up noise peaks that are not part of the horn sound (super super ad-hoc heuristic but it worked!)

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

def extract_rms_sequence(audio_chunk, sr, frame_ms=20, hop_ms=10):

    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    rms = librosa.feature.rms(
        y=audio_chunk,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False
    )[0]


    return rms

def symbolize_rms_quantiles(rms, n_bins=5):
    rms = np.asarray(rms)

    if len(rms) == 0:
        return np.array([], dtype=int)

    # If the RMS is flat or nearly flat, quantile binning can fail.
    # In that case, just return one repeated symbol.
    if np.allclose(rms, rms[0]):
        return np.zeros(len(rms), dtype=int)

    symbols = pd.qcut(
        rms,
        q=n_bins,
        labels=False,
        duplicates="drop"
    )

    return np.asarray(symbols, dtype=int)

def create_recurrence_matrix(symbols, m=1, d=1):
    
    symbols = np.asarray(symbols)

    n_states = len(symbols) - (m - 1) * d

    states = np.array([[symbols[i + k * d] for k in range(m)]for i in range(n_states)])

    recurrence_matrix = np.all(states[:, None, :] == states[None, :, :],axis=2).astype(int)

    return recurrence_matrix

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