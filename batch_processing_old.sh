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

# 🔁 FIXED: Windows → macOS paths
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
   
    exit 1

    while IFS= read -r file; do
        base=$(basename "$file")
        name="${base%.*}"
        expected="$BATCH/cropped_videos/${name}_cut.mp4"

        if [ ! -f "$expected" ]; then
            echo " Missing output for: $file"
            echo "$file" >> "$FAILED_LOG"
        fi

    done < current_batch.txt

    # step 5: move processed outputs back to OneDrive
    new_files=()

    for f in "$BATCH"/cropped_videos/*; do
        [ -f "$f" ] || continue
        cp "$f" "$DEST_PROCESSED/"
        new_files+=("$DEST_PROCESSED/$(basename "$f")")
    done

    # wait only on newly created files
    for f in "${new_files[@]}"; do
        wait_for_stable_file "$f"
    done

    sleep 20

    # NOTE: this is Windows-specific, will NOT work on Mac (but this could be improved however, most enterprise centered workflows are windows)
    # leaving it unchanged per your request
    find "$DEST_PROCESSED" -type f -exec attrib +U "{}" \;

    # step 6: clear local cache
    rm "$BATCH"/*.mp4 2>/dev/null
    rm -rf "$BATCH"/horn_audios
    rm -rf "$BATCH"/cropped_videos
    rm -f "$BATCH"/*.csv

done