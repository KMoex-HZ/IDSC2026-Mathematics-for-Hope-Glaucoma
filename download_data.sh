#!/bin/bash
set -e

echo "Checking for HYGD dataset..."
mkdir -p data/raw

ZIP_PATH="data/raw/hygd.zip"
DATASET_URL="https://physionet.org/static/published-projects/hillel-yaffe-glaucoma-dataset/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset-for-glaucoma-detection-1.0.0.zip"

if [ -f "$ZIP_PATH" ]; then
    echo "ZIP already found locally, skipping download..."
else
    echo "Downloading from PhysioNet..."
    wget -q --user=$PHYSIONET_USER --password=$PHYSIONET_PASS -O "$ZIP_PATH" "$DATASET_URL"
fi

echo "Extracting dataset..."
bsdtar -xvf "$ZIP_PATH" -C data/raw/

echo "Cleaning up zip file..."
rm "$ZIP_PATH"

echo "Data ready at data/raw/"