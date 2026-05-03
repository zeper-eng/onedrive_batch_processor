# OneDrive Batch Video Processor
Hybrid Bash + Python pipeline for processing OneDrive-hosted videos with batching and cloud file handling.

I did my best generalize it for public viewing but keep in mind many parts are "hardcoded" for our specific use-case.For example the band we are most interested in is hardcoded in terms of the horn reference sound matching.

# Background
This project came out of a real workflow problem. A teammate a while back had built scaffolding for an automated video cutting pipeline which compared the raw waveform of a horn sound audio signal (using librosa) and the waveform of the video we were trying to detect the horn sound in.

The goal was to crop it and keep the 10 seconds before the horn sound played and the 120 seconds after.

I refactored the pipeline in various ways including orchestrating a bash-side to scale processing of videos hosted on one-drive. I also engineered better features than comparing raw wavelengths, added a sliding window comparison component, and then trained a logistic regression using the same windows in the properly extracted videos compared to the improperly extracted videos to improve detection even further.

Before my method 68 out of 143 processed videos failed meaning a fail rate of about 47.6% of videos.

after my method 54 out of 406 processed videos failed meaning a fail rate of about 13.3% of videos.

After my MODEL 49 out of 407 processed videos failed meaning a fail rate of about 12.1%

# Feature Engineering and Model Incorporation

## The “better features” (FFT band + harmonics)
Look in:
- `vid_processing_modules/feature_extraction.py`
- `vid_processing_modules/vid_detection_utils.py`

The core idea is: instead of comparing raw waveforms, compare the *frequency content* of a 1-second window in a specific band.

What’s hardcoded here (and where):
- **Band-pass region**: `640–3400 Hz` (see band mask inside both `prepare_horn_template()` and `detect_horn()`).
- **Template**: a 1.0s clip from the reference horn file (`reference_audio/reference_event.wav`).
- **Harmonics**: multipliers `[1, 2, 3]` to build a “harmonic index set” from the strongest bins of the horn template.

What actually gets computed (features):
- `peak_match` — dot product between horn-template FFT bins and window FFT bins at the selected harmonic indices.
- `peak_energy` — total energy in those harmonic bins.
- `total_band_energy` — total energy in the whole 640–3400 Hz band.
- `concentration` — `peak_energy / total_band_energy`.
- `raw_score` — `peak_match * concentration` (this is the “no-model” score).

(See `extract_detector_features()` in `feature_extraction.py` and `extract_window_features()` in `vid_detection_utils.py`.)

## Sliding window detection
Look in:
- `vid_processing_modules/vid_detection_utils.py` (`sliding_window_detection()` and `detect_horn()`)

Detection scans the audio with:
- window size: **1.0s**
- hop size: **0.05s** (50ms)

It scores every window and keeps the best one (highest `raw_score` if no model, or highest model probability if a model is loaded).

## Model scoring (optional)
Look in:
- `vid_processing_modules/vid_detection_utils.py` (`score_window()`)

If `model_path` is provided, the model is loaded with `joblib` and `score_window()` switches from `raw_score` to:
- `model.predict_proba([[peak_match, peak_energy, total_band_energy, concentration]])`

The feature columns it expects are hardcoded as:
- `FEATURE_COLUMNS = ["peak_match", "peak_energy", "total_band_energy", "concentration"]`

The model path the bash runner uses is:
- `models/event_logistic_model.joblib` (set in `batch_vid_processing.sh`)

## Feature CSV generation (the stuff you train on)
Look in:
- `vid_processing_modules/feature_matrix_extraction.py`
- `vid_processing_modules/model_training.py`

Both scripts write a CSV with window-level rows and a `label` column:
- default output: `horn_training_features.csv`
- `WINDOWS` are hardcoded to `[(9.5, 10.5), (10.5, 11.5)]`
- sample rate is hardcoded to `TARGET_SR = 16000`
- duplicates get removed at the end

**Labeling**
- `feature_matrix_extraction.py` has an example hardcoded `FAILED_CUTS_new` list in-file.
- `model_training.py` imports `FAILED_CUTS_new` from a python module named `convenience` (not included here).

So if you actually run `model_training.py` in this public repo as-is, you’ll need a `convenience.py` alongside it that defines `FAILED_CUTS_new`, or just use `feature_matrix_extraction.py` as the “stand-in” example.

# Batch processing

If you’ve ever worked with OneDrive in a production setting, you already know the main issue: files aren’t always actually local. Between cloud-only states, inconsistent syncing, and large file sizes, just “looping over files” stops being reliable pretty quickly.

My solution was to build a pipeline that treats OneDrive like a semi-remote storage layer and processes files locally in controlled batches.

The workflow looks like this:

1. Force files to download locally (`attrib -U`)
2. Copy them into a local working directory (scratch space)
3. Process them in batches using the existing Python script
4. Move results back to OneDrive
5. Clean up local files to avoid storage issues
6. Log any failures for later inspection

The pipeline uses:
- Bash for orchestration, batching, and file/system operations
- Python for the actual signal-based video processing

This split keeps the system simple while still handling a pretty messy environment.

## The two “main” entrypoints
Look in:
- `batch_vid_processing.sh` (cropping pipeline)
- `batch_feature_extraction.sh` (feature extraction pipeline)

### 1) Batch video detection + crop (and push outputs back)
`batch_vid_processing.sh` does:
- collects videos with `find "$SRC" -type f -iname "*.mp4"`
- batches with `get_next_batch` (from `pipeline_utils.sh`)
- hydrates + copies each file into `local_batch/` via `local_download` (uses `attrib -U` + `wait_for_stable_file` + `cp` retries)
- runs the python entrypoint: `python vid_processing_modules/video_event_detection.py "$BATCH" "$REF" "$MODEL"`
- copies outputs from `$BATCH/cropped_videos/` into `$DEST_PROCESSED/` via `move_and_wait_outputs`
- unpins outputs and input files (`attrib +U`)
- logs anything missing via `identify_unprocessed_files` into `logs/failed_files.txt`
- cleans up `local_batch/` (mp4 + horn_audios + cropped_videos + csv)

Important:
- `attrib -U/+U` is Windows-specific. This is meant to be run in something like Git Bash on Windows / a Windows environment where those commands exist.

### 2) Batch feature extraction (append to a master CSV)
`batch_feature_extraction.sh` does:
- batches over `input_videos/`
- hydrates + copies to `local_batch/`
- runs: `python vid_processing_modules/feature_matrix_extraction.py "$BATCH" "$REF" "$TRAINING_CSV"`
- appends into `feature_sets/event_training_features_master.csv`

# Motivation
The goal here wasn’t just to “get it working,” but to make the workflow reliable when dealing with:

- cloud-backed file systems
- large datasets
- limited local storage

I also wanted to show how shell scripting can still be useful for system-level orchestration alongside Python.(and to demonstrate practical shell scripting for system-level orchestration)

# Notes

- The included Python script is a simplified version of the original. The underlying processing logic was built by a teammate rather than by me, but it still gives a clear sense of the overall workflow and how the pipeline was being used. The signal/video processing itself is not especially uncommon, the more relevant part of this project for me is the orchestration layer around it: handling OneDrive-hosted files, batching, local processing, and moving outputs back reliably.

- In theory there would be a virtual environment inside of `venv/` that would activate the proper python package installations needed to run modules such as librosa etc. (the bash scripts assume `source venv/Scripts/activate`)

- `ffmpeg` is required because cropping is done via a direct ffmpeg call (see `crop_video()` in `vid_detection_utils.py`).

- The file `failed_files.txt`, and the directory `local_batch/` are meant to simulate the kind of output you would get when running the pipeline.

*DISCLAIMER*: I AM NOT AN AUDIO EXPERT. THEESE FEATURES WHERE FORM WHAT I GATHERED WHERE COMMON PRACTICE AND WORKED FOR MY PURPOSES I AM SURE THERE ARE BETTER ALTERNATIVES.