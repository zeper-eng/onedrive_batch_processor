#!/bin/bash
# this ones just for functions that are used in multiple scripts

# Waits until a file stops changing size locally.
# NOTE: This does NOT verify OneDrive upload completion — it only ensures the file
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
        # -eq → numeric equality
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