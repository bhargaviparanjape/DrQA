#!/bin/sh
python script/train.py \
    --model-dir /projects/tir2/users/bvp/neulab/DrQA/rcmodels/MnemonicReader/models \
    --data-dir /projects/tir2/users/bvp/neulab/DrQA/rcmodels/MnemonicReader/data/datasets \
    --embed-dir /projects/tir2/users/bvp/embeddings/ \
    --model-name FULL_SQUAD \
