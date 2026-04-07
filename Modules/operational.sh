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
    attrib -U "$file" 2>/dev/null
    cp "$file" "$BATCH/"

done
}

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
