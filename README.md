
# OneDrive Batch Video Processor
Hybrid Bash + Python pipeline for processing OneDrive-hosted videos with batching and cloud file handling.

I did my best to generalize it for public viewing, but many parts are hardcoded for our specific use case. For example the band we are most interested in is hardcoded in terms of the horn reference sound matching.

# Background
This project came out of a real workflow problem. A teammate a while back had built scaffolding for an automated video cutting pipeline which used direct cross-correlation between the reference horn waveform and each video’s extracted audio. 
The goal was to crop it and keep the 10 seconds before the horn sound played and the 120 seconds after (using librosa).

I refactored the pipeline in various ways including orchestrating a bash-side to scale processing of videos hosted on OneDrive. I also engineered horn specific features, added a sliding window comparison component, and then trained a logistic regression using the same windows in the properly extracted videos compared to the improperly extracted videos to improve detection even further.

Before my method 194 out of 407 processed videos failed meaning a fail rate of about 47.6% of videos.

After incorporating sRQA and FFT/harmonic features, 33 out of 407 processed videos failed, bringing the fail rate down to approximately 8.1%. Eventually, this was dropped as other work took over, and we didn't have that many more videos to cut once the fail rate was already low enough but, it was a very interesting feature engineering sidequest for me personally!

# Feature Engineering and Model Incorporation

## The original “better features” (FFT band + harmonics)
The core idea was to compare the *frequency content* of each 1-second candidate window. The horn had a distinct frequency profile, so I inspected a spectrogram and identified a focused “band of interest” that captured the signal well: 640–3400 Hz.

The reference template was a 1-second horn clip from a public SFX database. In theory, this template could be replaced with another target sound, as long as the relevant frequency band and envelope are re-estimated.

The features:
- `peak_match` — dot product between the horn-template FFT bins and candidate-window FFT bins at the selected 1x, 2x, and 3x harmonic indices.
- `peak_energy` — total candidate-window energy at those selected 1x, 2x, and 3x harmonic indices.
- `raw_score` — the score used without a model: `peak_match * concentration`, rewarding windows that both match the horn template and concentrate energy in the expected harmonic bins (more on this below).


When theres no model:

- A sliding window approach is used where starting from the 0th second, and in .5s hops, overlapping windows are scaned and `raw_score` is calculated.
- The video with the best raw score is how our video cutting point is decided on.

I ran this on all videos and signifcantly reduced fail rate to about 15%. Now someone had to go through and check what worked and what didn't (which I also did).

## Model Training 

Once, I had checked I took a step back and thought, well if we label each with either a 0 (fail) or a 1 (pass) I essentially had a labeled dataset and if we are already calculating a score from new features, let's add two more:

- `total_band_energy` — total candidate-window energy across the full 640–3400 Hz band.
- `concentration` — proportion of band energy concentrated in the selected harmonic indices: `peak_energy / total_band_energy`.

And one of our PIs had recently submitted software on [sRQA](https://doi.org/10.64898/2026.03.31.715624), that motivated me to try sRQA-style features on the horn-detection problem:

![sRQA feature schematic](resources/just_matrices.png)

Thus the full set of feature I extracted from all videos became:
`["peak_match","peak_energy","total_band_energy","concentration","RR","DET","L","Lmax","DIV","ENTR","LAM","TT","Vmax","VENTR","MRT","RTE","NMPRT","TREND"]`

Then I trained a logistic regression on these features and re-ran the pipeline with a new option.

Rather than selecting the candidate window with the highest `raw_score`, the pipeline now extracted all 18 features for the top-scoring candidate windows and used the logistic regression to assign each window a probability of being a successful cut point. The highest-probability window was then selected as the horn-detection location used for cropping.

This allowed the detector to learn from examples of successful and unsuccessful cuts rather than relying entirely on manually designed scoring rules. In practice, the combination of FFT/harmonic features and sRQA features reduced the failure rate from roughly 15% to about 8%.

This is the core of the modelling and the batch processing follows below.

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

# Motivation
The goal here wasn’t just to “get it working,” but to make the workflow reliable when dealing with:

- cloud-backed file systems
- large datasets
- limited local storage

I also wanted to show how shell scripting can still be useful for system-level orchestration alongside Python and go about exploring feauture engineering myself one I was able to get my hands on some more projects a few months into my role!

# Notes

- The included Python script is a simplified version of the original. 

- In theory there would be a virtual environment inside of `venv/` that would activate the proper python package installations needed to run modules such as librosa etc. (the bash scripts assume `source venv/Scripts/activate`)

- `ffmpeg` is required because cropping is done via a direct ffmpeg call (see `crop_video()` in `vid_detection_utils.py`).

- The file `failed_files.txt`, and the directory `local_batch/` are meant to simulate the kind of output you would get when running the pipeline.

*DISCLAIMER*: I am not an audio expert. These features were based on methods I found were common practice and worked for my purpose. I am sure there are better alternatives.