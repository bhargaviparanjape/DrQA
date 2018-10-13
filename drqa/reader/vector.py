#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch
from ..selector.vector import vectorize as sent_selector_vectorize
from ..selector.vector import batchify_sentences as sent_selector_batchify
import numpy as np
import pdb

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

    sentence_boundaries = []
    counter = 0
    for sent in ex['sentences']:
        sentence_boundaries.append([counter, counter + len(sent)])
        counter += len(sent)

    # If Sentence Selector is turned on, then run the document through and get the top sentence
    if args.use_sentence_selector:
        sentence_lengths = [len(sent) for sent in ex['sentences']]
        max_length = max(sentence_lengths)
        sentences = torch.LongTensor([pad_single_seq([word_dict[w] for w in sent], max_length) for sent in ex['sentences']])

        if args.use_gold_sentence:
            # Use gold sentence
            if len(ex['gold_sentence_ids']) == 0:
                return []
            top_sentence = ex['gold_sentence_ids'][0]
        else:
            ex_batch = sent_selector_batchify([sent_selector_vectorize(ex, model.sentence_selector, single_answer)])
            top_sentence = model.sentence_selector.predict(ex_batch)[0][0]
            if len(ex['gold_sentence_ids']) > 0 and top_sentence not in ex['gold_sentence_ids']:
                return []
        # Extract top sentence and change ex["document"] accordingly
        document = torch.LongTensor([word_dict[w] for w in ex['sentences'][top_sentence]])
        ex['document'] = ex['sentences'][top_sentence]
        offset_subset = ex["offsets"][sentence_boundaries[top_sentence][0]:sentence_boundaries[top_sentence][1]]
        initial_offset = offset_subset[0][0]
        ex['offsets'] = [[t[0] - initial_offset, t[1] - initial_offset] for t in offset_subset]

        # Check if selected sentence contains any answer span
        # account for answers being in between the gold sentence
        window = sentence_boundaries[top_sentence]
        flag = True
        for answer in ex['answers']:
            if answer[0] >= window[0] and answer[1] < window[1]:
                new_start = answer[0] - window[0]
                new_end = answer[1] - window[0]
                flag = False
                break
            elif answer[0] >= window[0] and answer[1] < sentence_boundaries[top_sentence + 1][1]:
                new_start = answer[0] - window[0]
                new_end = window[1] - window[0] - 1
                flag = False
                break
        # Single Answer is False for development set
        if flag and single_answer == True:
            return []
        if not single_answer and len(ex['answers'])> 0:
            new_start = []
            new_end = []
            for top in ex["gold_sentence_ids"]:
                window = sentence_boundaries[top]
                for answer in ex['answers']:
                    if answer[0] >= window[0] and answer[1] < window[1]:
                        new_start.append(answer[0] - window[0])
                        new_end.append(answer[1] - window[0])
                    elif answer[0] >= window[0] and answer[1] < sentence_boundaries[top + 1][1]:
                        new_start.append(answer[0] - window[0])
                        new_end.append(answer[1] - window[1])



    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(len(ex['document']), len(feature_dict))
    else:
        features = None

    # f_{exact_match}
    if args.use_in_question:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if q_lemma and ex['lemma'][i] in q_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, features, question, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    # Find new locations in extracted Gold Sentence, if not, return a random sample of same length
    if single_answer:
        assert(len(ex['answers']) > 0)
        if args.use_sentence_selector:
            start = torch.LongTensor(1).fill_(new_start)
            end = torch.LongTensor(1).fill_(new_end)
        else:
            start = torch.LongTensor(1).fill_(ex['answers'][0][0])
            end = torch.LongTensor(1).fill_(ex['answers'][0][1])
    else:
        # Do same and send as list
        if len(ex['answers']) == 0:
            start = []
            end = []
        elif args.use_sentence_selector:
            start = new_start
            end = new_end
        else:
            start = [a[0] for a in ex['answers']]
            end = [a[1] for a in ex['answers']]

    return document, features, question, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1

    batch = [ex for ex in batch if len(ex) != 0]

    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(features[i])

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x2, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][3]):
            y_s = torch.cat([ex[3] for ex in batch])
            y_e = torch.cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids
