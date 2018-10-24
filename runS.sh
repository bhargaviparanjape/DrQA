#!/bin/sh
python scripts/selector/train.py \
	--embed-dir /projects/tir2/users/bvp/embeddings/ \
	--embedding-file glove.840B.300d.txt \
	--num-epochs 100 \
    --model-name SENT_SELECT_ADV_RANDOM1_SQuAD \
    --train-file SQuAD-v1.1-advrandom-train-processed-corenlp.txt \
	--data-dir /projects/tir2/users/bvp/neulab/DrQA/data/datasets/ \
	--model-dir /projects/tir2/users/bvp/neulab/DrQA/models/ \
    --pretrained /projects/tir2/users/bvp/neulab/DrQA/models/ORACLE_SQuAD_PATIENCE.mdl.mdl \
	--batch-size 16 \
	--display-iter 100 \
    --use-in-question True \
    --use-ner True \
    --use-lemma True \
    --use-tf True \
    --use-pos True \
    --tune-partial 1000 \
