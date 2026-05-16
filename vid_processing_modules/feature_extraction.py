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
    """
    Extract basic symbolic RQA (sRQA) features from a binary recurrence matrix.
    """

    R = np.asarray(R)
    N = R.shape[0]

    # ------------------------------------------------------------
    # Shannon entropy from empirical counts
    # ------------------------------------------------------------
    def shannon_entropy(values):
        if len(values) == 0:
            return 0.0

        _, counts = np.unique(values, return_counts=True)

        probs = counts / counts.sum()

        return -np.sum(probs * np.log(probs + 1e-12))

    # ------------------------------------------------------------
    # Diagonal line lengths
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Vertical line lengths
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Recurrence counts
    # ------------------------------------------------------------
    total_points = N * N
    recurrent_points = np.sum(R)

    RR = recurrent_points / total_points

    # ------------------------------------------------------------
    # Diagonal measures
    # ------------------------------------------------------------
    diag_points = np.sum(diag_lengths)

    DET = diag_points / recurrent_points if recurrent_points > 0 else 0

    L = np.mean(diag_lengths) if diag_lengths else 0

    Lmax = np.max(diag_lengths) if diag_lengths else 0

    DIV = 1 / Lmax if Lmax > 0 else 0

    ENTR = shannon_entropy(diag_lengths)

    # ------------------------------------------------------------
    # Vertical measures
    # ------------------------------------------------------------
    vert_points = np.sum(vert_lengths)

    LAM = vert_points / recurrent_points if recurrent_points > 0 else 0

    TT = np.mean(vert_lengths) if vert_lengths else 0

    Vmax = np.max(vert_lengths) if vert_lengths else 0

    VENTR = shannon_entropy(vert_lengths)

    # ------------------------------------------------------------
    # Recurrence times
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # TREND
    # ------------------------------------------------------------
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
    """ needed a docstring for this one to keep myself oriented although this will all be internal
    Compute a short-time RMS energy sequence for one audio chunk.

    This function does NOT return one RMS value for the whole chunk.
    Instead, it breaks the chunk into many small overlapping frames and
    computes one RMS value per frame. The result is a time-series describing
    how the signal's loudness/energy changes inside the chunk.

    For example, if audio_chunk is a 1-second window and sr is 44,100 Hz:

        frame_ms = 20  -> each RMS value is computed from 20 ms of audio
                          20 ms * 44,100 samples/sec = 882 samples

        hop_ms   = 10  -> after computing one RMS value, move forward 10 ms
                          10 ms * 44,100 samples/sec = 441 samples

    So the RMS sequence is computed like:

        RMS #1: samples from 0 ms   to 20 ms
        RMS #2: samples from 10 ms  to 30 ms
        RMS #3: samples from 20 ms  to 40 ms
        RMS #4: samples from 30 ms  to 50 ms
        ...

    Because the frame is 20 ms and the hop is 10 ms, adjacent frames overlap
    by 10 ms. This gives a smooth energy envelope rather than a single summary
    number.

    For a 1-second chunk with center=False, the approximate number of RMS values is:

        1 + floor((len(audio_chunk) - frame_length) / hop_length)

    With a 1-second chunk, 20 ms frames, and 10 ms hops, this is usually 99 values:

        1 + floor((1000 ms - 20 ms) / 10 ms)
        = 1 + 98
        = 99

    Each returned RMS value is:

        sqrt(mean(frame_samples ** 2))

    This is similar to a smoothed absolute-value/loudness curve. It removes the
    fast positive/negative oscillation of the waveform and keeps the slower
    energy shape inside the window, such as:

        quiet baseline -> horn onset -> rising energy -> sustained loud region

    That makes this sequence a better input for symbolic recurrence analysis
    than the raw waveform samples. The raw waveform mainly reflects rapid
    acoustic oscillation, while this RMS sequence reflects the within-window
    energy pattern that can be symbolized and used to build recurrence plots.

    Parameters
    ----------
    audio_chunk : np.ndarray
        A 1D audio waveform segment, usually one detector window such as
        a 1-second chunk.

    sr : int
        Sampling rate of the audio in samples per second.

    frame_ms : int or float, default=20
        Length of each small RMS frame in milliseconds. Larger values produce
        a smoother RMS sequence. Smaller values preserve more rapid changes.

    hop_ms : int or float, default=10
        Step size between consecutive RMS frames in milliseconds. Smaller values
        produce more RMS points and more overlap. Larger values produce fewer
        RMS points.

    Returns
    -------
    rms : np.ndarray
        A 1D array of RMS energy values. For a 1-second chunk with the default
        20 ms frame and 10 ms hop, this will usually contain 99 values.

    """

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
    """
    Create a symbolic recurrence matrix from a sequence of symbols.

    Each row/column represents a time-indexed symbolic state derived from
    the RMS envelope of an audio window.

    Parameters
    ----------
    symbols : array-like
        Symbolized RMS sequence, e.g. [1, 2, 3, 4, 2, ...].

    m : int, default=1
        Embedding dimension / word length.

        m=1 compares individual symbols:
            [symbol[i]]

        m=3 compares 3-symbol patterns:
            [symbol[i], symbol[i+d], symbol[i+2d]]

    d : int, default=1
        Time delay / spacing between symbols inside each embedded word.

        d=1 uses adjacent symbols:
            [symbol[i], symbol[i+1], symbol[i+2]]

        d=2 skips every other symbol:
            [symbol[i], symbol[i+2], symbol[i+4]]

        In your RMS setup, if hop_ms=10:
            d=1 means 10 ms spacing
            d=2 means 20 ms spacing
            d=5 means 50 ms spacing


    Returns
    -------
    recurrence_matrix : np.ndarray
        Square binary matrix where recurrence_matrix[i, j] = 1 if the symbolic
        state/pattern at time i equals the symbolic state/pattern at time j.
    """
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