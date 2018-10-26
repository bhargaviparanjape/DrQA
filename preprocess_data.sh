#!/bin/sh
python -u scripts/selector/preprocess.py \
	--data_dir /projects/tir2/users/bvp/neulab/DrQA/data/datasets/ \
	--out_dir /projects/tir2/users/bvp/neulab/DrQA/data/datasets/ \
	--split SQuAD-v1.1-advrandom-train \
