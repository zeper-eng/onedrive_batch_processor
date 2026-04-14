###########################################################
#                  GLOBAL CONFIG                          #
###########################################################

#virtual environment setup and imports
source venv/Scripts/activate
source Modules/bash_functions.sh
source Modules/operational.sh

#global variables
REF="Horn_Reference_sound.wav"
FAILED_LOG="failed_files.txt" > "$FAILED_LOG"
SRC="/Users/luis/OneDrive/Columbia University Irving Medical Center/Remote DAYC-2 and Maternal Cognition - Horncut_Highchair_Videos/Testing"
DEST_PROCESSED="/Users/luis/OneDrive/Columbia University Irving Medical Center/Remote DAYC-2 and Maternal Cognition - Horncut_Highchair_Videos/Testing/Testing_output"
BATCH=~/Projects/Horn_Task/local_batch
BATCH_SIZE=2

# create necessary directories and validate input scripts before processing
mkdir -p "$BATCH"
mkdir -p "$DEST_PROCESSED"

validate_input_scripts "$REF" "Modules/Horn_detection_pipeline_luis.py" "$FAILED_LOG" "$SRC" "$DEST_PROCESSED"

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
    python Modules/Horn_detection_pipeline_luis.py "$BATCH" "$REF"

    identify_unprocessed_files batch

    # move processed outputs back to OneDrive
    new_files=()
    move_and_wait_outputs new_files

    sleep 20 #ja little extra-extra buffer room just incase

    # NOTE: this is Windows-specific, will NOT work on Mac (but this could be improved however, most enterprise centered workflows are windows)
    find "$DEST_PROCESSED" -type f -exec attrib +U "{}" \;

    # clear local cache
    rm "$BATCH"/*.mp4 2>/dev/null
    rm -rf "$BATCH"/horn_audios
    rm -rf "$BATCH"/cropped_videos
    rm -f "$BATCH"/*.csv

done