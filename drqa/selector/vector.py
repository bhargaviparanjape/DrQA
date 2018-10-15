#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch


def pad_single_seq(seq, max_len, pad_token = 0):
    seq += [pad_token for i in range(max_len - len(seq))]
    return seq


def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    sentence_lengths = [len(sent) for sent in ex['sentences']]
    max_length = max(sentence_lengths)
    sentences = torch.LongTensor([pad_single_seq([word_dict[w] for w in sent], max_length) for sent in ex['sentences']])

    # Prepare other features also by sentence
    sentence_boundaries = []
    counter = 0
    for sent in ex['sentences']:
        sentence_boundaries.append([counter, counter + len(sent)])
        counter += len(sent)
    poses = [ex['pos'][sentence_boundaries[i][0]: sentence_boundaries[i][1]]
             for i in range(len(ex["sentences"]))]
    ners = [ex['ner'][sentence_boundaries[i][0]: sentence_boundaries[i][1]]
             for i in range(len(ex["sentences"]))]
    lemmas = [ex['lemma'][sentence_boundaries[i][0]: sentence_boundaries[i][1]]
             for i in range(len(ex["sentences"]))]

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['sentences']), max_length, len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        ## All sent will have equal length as they are padded
        for j, sent in enumerate(ex["sentences"]):
            for i in range(len(sent)):
                if sent[i] in q_words_cased:
                    features[j][i][feature_dict['in_question']] = 1.0
                if ex['document'][i].lower() in q_words_uncased:
                    features[j][i][feature_dict['in_question_uncased']] = 1.0
                if q_lemma and lemmas[j][i] in q_lemma:
                    features[j][i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for j, pos in enumerate(poses):
            for i, w in enumerate(pos):
                f = 'pos=%s' % w
                # pos=UNK not in any of the POS features, will return 0
                if f in feature_dict:
                    features[j][i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for j, ner in enumerate(ners):
            for i, w in enumerate(ner):
                f = 'ner=%s' % w
                if f in feature_dict:
                    features[j][i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        for j, sent in enumerate(ex["sentences"]):
            counter = Counter([w.lower() for w in sent])
            ## Original length of sentence
            l = len(sent)
            for i, w in enumerate(sent):
                features[j][i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return sentences, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    if single_answer:
        assert(len(ex['answers']) > 0)
        start = torch.LongTensor(1).fill_(ex['answers'][0][0])
        end = torch.LongTensor(1).fill_(ex['answers'][0][1])
        gold_id = torch.LongTensor(1).fill_(ex['gold_sentence_ids'][0])
    else:
        start = [a[0] for a in ex['answers']]
        end = [a[1] for a in ex['answers']]
        gold_id = ex['gold_sentence_ids']

    return sentences, sentence_lengths, features, question, gold_id, ex['id']

def batchify_sentences(batch):
    NUM_INPUTS = 4
    NUM_TARGETS = 1
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    doc_lengths = [ex[1] for ex in batch]
    features = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(1) for d in docs])
    sentence_lengths = [d.size(0) for d in docs]
    max_sentences = max(sentence_lengths)
    x1 = torch.LongTensor(len(docs), max_sentences, max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_sentences, max_length).fill_(1)
    x1_sent_mask = torch.FloatTensor(len(docs), max_sentences).zero_()

    for i, d in enumerate(docs):
        x1[i, :d.size(0), :d.size(1)].copy_(d)
        for j, l in enumerate(doc_lengths[i]):
            x1_mask[i, j, :l].fill_(0)
        for j in range(sentence_lengths[i], max_sentences):
            ## put some UNK token into the padded sentences and make thier length 1
            x1[i, j, 0].fill_(1)
            x1_mask[i, j, 0].fill_(0)
            ## actual padding over sentences to be used in logsoftmax
        x1_sent_mask[i, :sentence_lengths[i]].fill_(1)
        ## add UNK to the empty dialogues

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, None, x1_mask, x1_sent_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][4]):
            # y_s = torch.cat([ex[3] for ex in batch])
            # y_e = torch.cat([ex[4] for ex in batch])
            y_g = torch.cat([ex[4] for ex in batch])
        else:
            # y_s = [ex[3] for ex in batch]
            # y_e = [ex[4] for ex in batch]
            y_g = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, None, x1_mask, x1_sent_mask, x2, x2_mask, y_g, ids

def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 4
    NUM_TARGETS = 1
    NUM_EXTRA = 1

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    doc_lengths = [ex[1] for ex in batch]
    features = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(1) for d in docs])
    sentence_lengths = [d.size(0) for d in docs]
    max_sentences = max(sentence_lengths)
    x1 = torch.LongTensor(len(docs), max_sentences, max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_sentences, max_length).fill_(1)
    x1_sent_mask = torch.FloatTensor(len(docs), max_sentences).zero_()
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_sentences, max_length, features[0].size(2))
    for i, d in enumerate(docs):
        x1[i, :d.size(0), :d.size(1)].copy_(d)
        for j, l in enumerate(doc_lengths[i]):
            x1_mask[i, j, :l].fill_(0)
        for j in range(sentence_lengths[i], max_sentences):
            ## put some UNK token into the padded sentences and make thier length 1
            x1[i, j, 0].fill_(1)
            x1_mask[i,j,0].fill_(0)
            ## actual padding over sentences to be used in logsoftmax
        x1_sent_mask[i, :sentence_lengths[i]].fill_(1)
        ## add UNK to the empty dialogues
        if x1_f is not None:
            ## not handled
            x1_f[i, :d.size(0), :d.size(1), :].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x1_sent_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][4]):
            # y_s = torch.cat([ex[3] for ex in batch])
            # y_e = torch.cat([ex[4] for ex in batch])
            y_g = torch.cat([ex[4] for ex in batch])
        else:
            # y_s = [ex[3] for ex in batch]
            # y_e = [ex[4] for ex in batch]
            y_g = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x1_sent_mask, x2, x2_mask, y_g, ids
