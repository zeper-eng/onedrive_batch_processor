#End-to-end video processing pipeline:
#
#- Extracts audio from video files
#- Detects horn event via cross-correlation
#- Crops video around detected event
#- Outputs processed clips and summary CSV
#
#Designed for batch processing of large, cloud-hosted datasets.

#!/usr/bin/env bash
set -euo pipefail

###########################################################
#                  GLOBAL CONFIG                          #
###########################################################

#virtual environment for python packages setup and imports
source venv/Scripts/activate
source batch_processing_modules/pipeline_utils.sh

#global variables
PROJECT_DIR="$HOME/Projects/video_event_detection"
REF="$PROJECT_DIR/reference_audio/reference_event.wav"
FAILED_LOG="$PROJECT_DIR/logs/failed_featurextraction_files.txt"
SRC="$PROJECT_DIR/input_videos"
TRAINING_CSV="$PROJECT_DIR/feature_sets/horn_training_features_master.csv"
BATCH="$PROJECT_DIR/local_batch"
BATCH_SIZE=20

mkdir -p "$(dirname "$FAILED_LOG")"
> "$FAILED_LOG"

# create necessary directories and validate input scripts before processing
mkdir -p "$BATCH"
mkdir -p "$(dirname "$TRAINING_CSV")"

validate_input_scripts "$REF" "vid_processing_modules/feature_matrix_creation.py" "$FAILED_LOG" "$SRC" 

###########################################################
#                         RUNNER                          #
###########################################################

# Build master file list (relative paths)
mapfile -t files < <(find "$SRC" -type f -iname "*.mp4")
echo "Found ${#files[@]} videos."
echo "Writing features to: $TRAINING_CSV"

while [ ${#files[@]} -gt 0 ]; do
    get_next_batch files batch "$BATCH_SIZE"

    # Clear old local files before copying next batch
    rm -f "$BATCH"/*.mp4 2>/dev/null

    # Copy/sync current batch locally
    local_download batch

    # Extract features and append to master CSV
    python vid_processing_modules/feature_matrix_creation.py "$BATCH" "$REF" "$TRAINING_CSV"

    # Delete local copies after features are saved
    rm -f "$BATCH"/*.mp4 2>/dev/null

    echo "Remaining videos: ${#files[@]}"
done

echo "Done."
echo "Master CSV:"
echo "$TRAINING_CSV"