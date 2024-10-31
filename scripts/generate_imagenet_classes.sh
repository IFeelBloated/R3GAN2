#!/bin/bash

set -e

# Path to the network snapshot and output directory
NETWORK_SNAPSHOT="$1"
OUTPUT_DIR="$2"

# Loop through each class ID (0 to 999)
for CLASS_ID in $(seq 0 999); do
    # Define the output directory for this class
    CLASS_DIR="${OUTPUT_DIR}/class_${CLASS_ID}"

    # Create the directory if it doesn't exist
    mkdir -p "$CLASS_DIR"

    # Define seed range for 1000 images
    SEED_START=$((CLASS_ID * 50 + 1))
    SEED_END=$((CLASS_ID * 50 + 50))

    # Generate images for this class
    CUDA_VISIBLE_DEVICES=0 python gen_images.py --network "$NETWORK_SNAPSHOT" --outdir "$CLASS_DIR" --seeds "${SEED_START}-${SEED_END}" --class "$CLASS_ID"
done

echo "Image generation complete."
