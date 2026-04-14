"""
End-to-end video processing pipeline:

- Extracts audio from video files
- Detects horn event via cross-correlation
- Crops video around detected event
- Outputs processed clips and summary CSV

Designed for batch processing of large, cloud-hosted datasets.
"""

###########################################################
#                  GLOBAL CONFIG                          #
###########################################################

#virtual environment for python packages setup and imports
source path/to/venv/bin/activate
source Modules/pipeline_utils.sh

#global variables
REF="Horn_Reference_sound.wav"
FAILED_LOG="failed_files.txt" > "$FAILED_LOG"
SRC="path/to/work/videos"
DEST_PROCESSED="path/to/work/processed_output"
BATCH="path/to/local_batch_workspace"
BATCH_SIZE=5

# create necessary directories and validate input scripts before processing
mkdir -p "$BATCH"
mkdir -p "$DEST_PROCESSED"

validate_input_scripts "$REF" "Modules/video_event_detection.py" "$FAILED_LOG" "$SRC" "$DEST_PROCESSED"

###########################################################
#                         RUNNER                          #
###########################################################

# Build master file list (relative paths)
mapfile -t files < <(find "$SRC" -type f -iname "*.mp4")

while [ ${#files[@]} -gt 0 ]; do

    # Get next batch of files to process
    get_next_batch files batch "$BATCH_SIZE"
    
    # Download locally by forcing OneDrive to sync them
    local_download batch
    python Modules/video_event_detection.py "$BATCH" "$REF"

    identify_unprocessed_files batch

    # move processed outputs back to source destination
    new_files=()
    move_and_wait_outputs new_files

    sleep 20 # little extra buffer just in case

    # NOTE: this is Windows-specific, will NOT work on Mac
    find "$DEST_PROCESSED" -type f -exec attrib +U "{}" \;

    # clear local cache
    rm "$BATCH"/*.mp4 2>/dev/null
    rm -rf "$BATCH"/horn_audios
    rm -rf "$BATCH"/cropped_videos
    rm -f "$BATCH"/*.csv

done