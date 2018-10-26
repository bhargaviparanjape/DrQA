#!/bin/sh
python scripts/reader/train.py \
	--embed-dir /projects/tir2/users/bvp/embeddings/ \
	--embedding-file glove.840B.300d.txt \
	--num-epochs 100 \
	--data-dir /projects/tir2/users/bvp/neulab/DrQA/data/datasets/ \
    --train-file SQuAD-v1.1-advrandom-train-processed-corenlp.txt \
	--model-dir /projects/tir2/users/bvp/neulab/DrQA/models/ \
    --model-name FULL_SQuAD_ADV_RANDOM1 \
	--batch-size 32 \
    --valid-metric f1 \
	--display-iter 100 \
    --use-in-question True \
    --use-ner True \
    --use-lemma True \
    --use-tf True \
    --use-pos True \
    --tune-partial 1000 \
