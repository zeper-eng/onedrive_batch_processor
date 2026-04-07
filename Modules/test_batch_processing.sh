# === GLOBAL CONFIG ===
source venv/Scripts/activate
which python
python --version
REF="Horn_Reference_sound.wav"

###########################################################
#block to find the reference file in the current directory
if [ ! -f "$REF" ]; then
    echo "❌ ERROR: Reference file not found: $REF"
    echo "Current directory: $(pwd)"
    echo "Available files:"
    ls -lh
    exit 1
else
    echo "✅ Found reference file: $REF"
    ls -lh "$REF"
fi
###########################################################


# === CONFIG ===
SRC="/c/Users/lp3200/Columbia University Irving Medical Center/Remote DAYC-2 and Maternal Cognition - Highchair"
DEST_PROCESSED="C:\Users\lp3200\Columbia University Irving Medical Center\Remote DAYC-2 and Maternal Cognition - Horncut_Highchair_Videos"
BATCH=~/Projects/Horn_Task/local_batch

mkdir -p "$BATCH"
mkdir -p "$DEST_PROCESSED"
source code/bash_functions.sh

# === RUNNER (TEST MODE: 2nd file only) ===

# build file list
find "$SRC" -type f -iname "*.mp4" > file_list.txt

# grab ONLY the 2nd file
sed -n '3p' file_list.txt > current_batch.txt

echo "Processing file:"
cat current_batch.txt

# step 3: copy + force download
while IFS= read -r file; do
    attrib -U "$file" 2>/dev/null
    cp "$file" "$BATCH/"
done < current_batch.txt

# step 4: process
python code/Horn_detection_pipeline_luis.py \
    "$BATCH" \
    Horn_Reference_sound.wav

# step 5: move processed outputs back to OneDrive
new_files=()

for f in "$BATCH"/cropped_videos/*; do
    [ -f "$f" ] || continue
    cp "$f" "$DEST_PROCESSED/"
    new_files+=("$DEST_PROCESSED/$(basename "$f")")
done

# wait only on new files
for f in "${new_files[@]}"; do
    wait_for_stable_file "$f"
done

sleep 10

# unpin (cloud-only)
find "$DEST_PROCESSED" -type f -exec attrib +U "{}" \;

# cleanup
rm "$BATCH"/*.mp4 2>/dev/null
rm -rf "$BATCH"/horn_audios
rm -rf "$BATCH"/cropped_videos
rm -f "$BATCH"/*.csv

