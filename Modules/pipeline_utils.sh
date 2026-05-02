# function that checks if input scripts are present
#
#Arguments:
# $1 - list of input script paths (passed as arguments)
#
validate_input_scripts(){
    local scripts=("$@") #$@ captures all arguments passed to the function as an array
    for script in "${scripts[@]}"; do
        if [ ! -e "$script" ]; then
            echo "Error: Required file '$script' not found."
            exit 1
        else
            echo "Found required file: $script"

        fi
    done
}

# function to get the next batch of files to process
#
#Arguments:
# $1 - name of the main files array (passed by reference)
# $2 - name of the batch output array (passed by reference)
# $3 - batch size (number of files to include in each batch)

get_next_batch() {
    local -n files_ref=$1   # reference to main files array
    local -n batch_ref=$2   # reference to batch output
    local size=$3           # batch size

    batch_ref=( "${files_ref[@]:0:size}" )
    files_ref=( "${files_ref[@]:size}" )
}

# function to download the current batch of files locally by forcing OneDrive to sync them
#
#Arguments:
# $1 - array containing the file paths to download (passed by reference)
# 

local_download() {
    local -n files_ref="$1"
    #echo "Downloading batch of ${#files_ref[@]} files locally..."
    #printf '%s\n' "${files_ref[@]}"
    #
    #exit 1 #temporary exit to prevent accidental execution while testing
    for file in "${files_ref[@]}"; do
        echo "Downloading/copying: $file"

        attrib -U "$file" 2>/dev/null || true

        for attempt in {1..10}; do
            if cp "$file" "$BATCH/" 2>/dev/null; then
                echo "Copied: $(basename "$file")"
                break
            fi

            echo "Copy failed, retry $attempt/10..."
            sleep 3
        done

        dest="$BATCH/$(basename "$file")"
        if [ ! -s "$dest" ]; then
            echo "FAILED COPY: $file"
            echo "$file" >> "$FAILED_LOG"
        fi

        done
    echo "Batch now contains:"
    ls -lh "$BATCH"
}

# function to download the current batch of files locally by forcing OneDrive to sync them
#
#Arguments:
# $1 - array containing the file paths to download (passed by reference)
# 

identify_unprocessed_files() {
    local -n batch_ref="$1"
    
    for file in "${batch_ref[@]}"; do
        base=$(basename "$file") #extract the file name from the path
        name="${base%.*}" #remove the file extension to get the base name
        expected="$BATCH/cropped_videos/${name}_cut.mp4"

    if [ ! -f "$expected" ]; then
        echo "Missing output for: $file"
        echo "$file" >> "$FAILED_LOG"
    fi
done

}

# Waits until a file stops changing size locally.
# NOTE: This does NOT verify OneDrive upload completion it only ensures the file
# is fully written before downstream actions (e.g., syncing or unpinning).

wait_for_stable_file() {

    # $1 = first argument passed into function
    # store it locally so we don’t mess with global variables
    local file="$1"

    # track previous file size (initialize to 0)
    local prev_size=0

    # infinite loop, will break manually when condition is met
    while true; do

        # get current file size in bytes
        # stat -c%s → prints file size
        # 2>/dev/null → suppress errors (e.g., file not ready yet)
        size=$(stat -c%s "$file" 2>/dev/null)

        # compare current size to previous size
        if [ "$size" -eq "$prev_size" ]; then
            # if size hasn’t changed → file is stable → exit loop
            break
        fi

        # update previous size for next iteration
        prev_size=$size

        # wait 5 seconds before checking again
        sleep 5
    done
}


move_and_wait_outputs() {
    local -n new_files_ref=$1   # pass array by reference

    new_files_ref=()

    for f in "$BATCH"/cropped_videos/*; do
        [ -f "$f" ] || continue
        dest="$DEST_PROCESSED/$(basename "$f")"
        cp "$f" "$dest"
        new_files_ref+=("$dest")
    done

    # wait only on newly created files
    for f in "${new_files_ref[@]}"; do
        wait_for_stable_file "$f"
    done
}