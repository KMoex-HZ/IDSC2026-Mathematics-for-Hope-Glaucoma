#!/bin/bash
# download_data.sh
# Downloads the HYGD dataset from PhysioNet and extracts it into data/raw/

set -e  # Exit immediately if any command fails

echo "Downloading HYGD dataset from PhysioNet..."
mkdir -p data/raw

DATASET_URL="https://physionet.org/static/published-projects/hillel-yaffe-glaucoma-dataset/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset-for-glaucoma-detection-1.0.0.zip"
ZIP_PATH="data/raw/hygd.zip"

# FIX: Separate wget download and extraction into two steps.
# Original used a broken pipe (wget -qO url | bsdtar) which fails because
# wget -qO requires a filename argument, not a URL as the -O target.
wget -q -O "$ZIP_PATH" "$DATASET_URL"

echo "Extracting dataset..."
bsdtar -xvf "$ZIP_PATH" -C data/raw/

echo "Cleaning up zip file..."
rm "$ZIP_PATH"

echo "Data ready at data/raw/"