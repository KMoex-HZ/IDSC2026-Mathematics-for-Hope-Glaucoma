#!/bin/bash
echo "Downloading HYGD dataset from PhysioNet..."
mkdir -p data/raw
wget -qO- https://physionet.org/static/published-projects/hillel-yaffe-glaucoma-dataset/hillel-yaffe-glaucoma-dataset-hygd-a-gold-standard-annotated-fundus-dataset-for-glaucoma-detection-1.0.0.zip | bsdtar -xvf- -C data/raw/
echo "Data ready for processing!"