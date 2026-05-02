#End-to-end video processing pipeline:
#
#- Extracts audio from video files
#- Detects horn event via cross-correlation
#- Crops video around detected event
#- Outputs processed clips and summary CSV
#
#Designed for batch processing of large, cloud-hosted datasets.


###########################################################
#                  GLOBAL CONFIG                          #
###########################################################

#virtual environment for python packages setup and imports
source venv/Scripts/activate
source batch_processing_modules/pipeline_utils.sh
source batch_processing_modules/convenience.sh #this has my array of broken files

#global variables
PROJECT_DIR="$HOME/Projects/video_event_detection"
REF="$PROJECT_DIR/reference_audio/reference_event.wav"
MODEL="$PROJECT_DIR/models/event_logistic_model.joblib"
FAILED_LOG="$PROJECT_DIR/logs/failed_files.txt"
SRC="$PROJECT_DIR/input_videos"
DEST_PROCESSED="$PROJECT_DIR/processed_event_clips"
BATCH="$PROJECT_DIR/local_batch"

BATCH_SIZE=5

# create necessary directories and validate input scripts before processing
mkdir -p "$BATCH"
mkdir -p "$DEST_PROCESSED"

validate_input_scripts "$MODEL" "$REF" "vid_processing_modules/video_event_detection.py" "$FAILED_LOG" "$SRC" "$DEST_PROCESSED"

###########################################################
#                         RUNNER                          #
###########################################################

# Build master file list (relative paths)
mapfile -t files < <(find "$SRC" -type f -iname "*.mp4") #for now i went recursive so top then bototom cuz not all were edited but this can change

while [ ${#files[@]} -gt 0 ]; do

    # Get next batch of files to process
    get_next_batch files batch "$BATCH_SIZE"
    
    # Download locally by forcing OneDrive to sync them
    local_download batch
    python vid_processing_modules/video_event_detection.py "$BATCH" "$REF" "$MODEL"

    # move processed outputs back to source destination
    new_files=()
    move_and_wait_outputs new_files

    sleep 20 # little extra buffer just in case

    # NOTE: this is Windows-specific, will NOT work on Mac
    find "$DEST_PROCESSED" -type f -exec attrib +U "{}" \;
    identify_unprocessed_files batch
    
    for file in "${batch[@]}"; do
        echo "Unpinning: $file"
        attrib +U "$file" 2>/dev/null || true
    done
    # clear local cache
    rm "$BATCH"/*.mp4 2>/dev/null
    rm -rf "$BATCH"/horn_audios
    rm -rf "$BATCH"/cropped_videos
    rm -f "$BATCH"/*.csv

done