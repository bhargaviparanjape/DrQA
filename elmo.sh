#!/bin/sh
python -u scripts/reader/prepare_elmo.py \
    --input-file /projects/tir2/users/bvp/neulab/DrQA/resources/SQuAD-v1.1-train_elmo \
    --output-file /projects/tir2/users/bvp/neulab/DrQA/resources/SQuAD-v1.1-train \
    --workers 10
