#!/bin/zsh

pip3 install -r requirements.txt
python3 preprocess.py \
   --bucket postal-parser-28 \
   --region europe-west1 \
   --input_subfolder unprocessed-data/testdata/
