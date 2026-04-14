# OneDrive Batch Video Processor
Hybrid Bash + Python pipeline for processing OneDrive-hosted videos with batching and cloud file handling.

# Background
This project came out of a real workflow problem. A teammate had already built a Python script that could detect and cut segments from videos based on a reference audio signal (using librosa). It worked well, but only for small, manually handled batches.

I was tasked with scaling that up to a much larger set of videos stored in OneDrive.

If you’ve ever worked with OneDrive in a production setting, you already know the main issue: files aren’t always actually local. Between cloud-only states, inconsistent syncing, and large file sizes, just “looping over files” stops being reliable pretty quickly.

# Approach

My solution was to build a pipeline that treats OneDrive like a semi-remote storage layer and processes files locally in controlled batches.

The workflow looks like this:

1. Force files to download locally (attrib -U)
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

I also wanted to clean up and modularize the original script into something reusable and easier to reason about, while showing how shell scripting can still be useful for system-level orchestration alongside Python.(and to demonstrate practical shell scripting for system-level orchestration)

# Notes

- The included Python script is a simplified version of the original. The underlying processing logic was built by a teammate rather than by me, but it still gives a clear sense of the overall workflow and how the pipeline was being used. The signal/video processing itself is not especially uncommon, the more relevant part of this project for me is the orchestration layer around it: handling OneDrive-hosted files, batching, local processing, and moving outputs back reliably.

- In theory there would be a virtual environment inside of venv/ that would activate the proper python package installations needed to run modules such as librosa etc.

- The file failed_files.txt, and the directory batch/ are also meant to simulate the kind of output you would get when running the pipeline