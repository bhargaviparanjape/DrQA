#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Functions for putting examples into torch format."""

from collections import Counter
import torch
from drqa.selector.vector import vectorize as sent_selector_vectorize
from drqa.selector.vector import batchify as sent_selector_batchify

def pad_single_seq(seq, max_len, pad_token = 0):
    seq += [pad_token for i in range(max_len - len(seq))]
    return seq

class sentence_batchifier():
    def __init__(self, model, single_answer):
        self.args = model.args
        self.model = model
        self.single_answer = single_answer

    def batchify(self, batch):
        (batch_sentence_boundaries, batch_offsets, batch_start, batch_end) = ([b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch], [b[3] for b in batch])
        # x1, x1_f, x1_mask, x1_sent_mask, x2, x2_mask, y_g, ids
        sentence_selector_batch = sent_selector_batchify([b[-1] for b in batch])
        batch_output, ids = sentence_selector_batch[:-1], sentence_selector_batch[-1]

        if self.args.use_gold_sentence:
            # Use gold sentence
            if len(batch_output) == 6:
                # Condition never hits because supervision always exists
                return []
            top_sentences = batch_output[6]
        else:
            if self.args.dynamic_selector:
                top_sentences = self.model.sentence_selector.predict(batch_output, use_threshold=self.args.selection_threshold)[0]
            else:
                top_sentences = self.model.sentence_selector.predict(batch_output)

        # Trim the batch based on selected sentences : torchy way (no support for dynamic context, no support for  k = 3)
        selected_questions = []
        selected_question_masks = []
        selected_ids = []
        selected_sentences = []
        selected_features = []
        selected_mask = []
        selected_offsets = []
        new_starts = []
        new_ends = []

        gold_sentences = batch_output[-1]
        for i in range(batch_output[0].shape[0]):
            # First Locate answer, if not present, then skip while training
            flag = True
            sentence_boundaries = batch_sentence_boundaries[i]
            offsets = batch_offsets[i]
            window = sentence_boundaries[top_sentences[i][0]]

            # Gold starts and ends (will be lists during inference)
            if torch.is_tensor(batch_start[i]):
                starts = batch_start[i].data.numpy().tolist()
                ends = batch_end[i].data.numpy().tolist()
            else:
                starts = batch_start[i]
                ends = batch_end[i]


            for answer in zip(starts, ends):
                if answer[0] >= window[0] and answer[1] < window[1]:
                    new_start = answer[0] - window[0]
                    new_end = answer[1] - window[0]
                    flag = False
                    break
                elif answer[0] >= window[0] and answer[1] < sentence_boundaries[top_sentences[i][0] + 1][1] and answer[0] < window[1]:
                    new_start = answer[0] - window[0]
                    new_end = window[1] - window[0] - 1
                    flag = False
                    break
                # Single Answer is False for development set
                # Logic of Matt Gardner comes here (Zack's extension to make to oracle better)
            if flag and self.single_answer == True:
                continue
            # At dev time, give all possible changed starts and ends
            if not self.single_answer and len(starts) > 0:
                new_start = []
                new_end = []
                for top in gold_sentences[i]:
                    window = sentence_boundaries[top]
                    for answer in zip(starts, ends):
                        if answer[0] >= window[0] and answer[1] < window[1]:
                            new_start.append(answer[0] - window[0])
                            new_end.append(answer[1] - window[0])
                        elif answer[0] >= window[0] and answer[1] < sentence_boundaries[top + 1][1]:
                            new_start.append(answer[0] - window[0])
                            new_end.append(answer[1] - window[1])

            new_starts.append(new_start)
            new_ends.append(new_end)
            selected_questions.append(batch_output[4][i].unsqueeze(0))
            selected_question_masks.append(batch_output[5][i].unsqueeze(0))
            selected_ids.append(ids[i])
            selected_sentences.append(batch_output[0][i][top_sentences[i][0], :].unsqueeze(0))
            selected_features.append(batch_output[1][i][top_sentences[i][0], :].unsqueeze(0))
            selected_mask.append(batch_output[2][i][top_sentences[i][0], :].unsqueeze(0))
            selected_offsets.append(offsets[sentence_boundaries[top_sentences[i][0]][0]:sentence_boundaries[top_sentences[i][0]][1]])

        ## How will sentence mask be used here : doesnt have to be since the model will never pick anything that is in the mask
        question = torch.cat(selected_questions, dim=0)
        question_mask = torch.cat(selected_question_masks, dim=0)
        document = torch.cat(selected_sentences, dim=0)
        document_mask = torch.cat(selected_mask, dim=0)
        document_features = torch.cat(selected_features, dim=0)

        ## Trim if batch size has been reduced (do this to exactly reproduce previous results)

        ## Depending on single answer either torchify starts and ends or maintain them as lists
        if self.single_answer:
            y_s = torch.LongTensor(new_starts)
            y_e = torch.LongTensor(new_ends)
        else:
            y_s = new_starts
            y_e = new_ends

        return document, document_features, document_mask, question, question_mask, y_s, y_e, selected_offsets, selected_ids

# def vectorize(ex, model, single_answer=False):
#     """Torchify a single example."""
#     args = model.args
#     word_dict = model.word_dict
#     char_dict = model.char_dict
#     feature_dict = model.feature_dict
#
#     # Index words
#     document = torch.LongTensor([word_dict[w] for w in ex['document']])
#     document_char = torch.LongTensor([char_dict[w] for w in ex['document_char']])
#     question = torch.LongTensor([word_dict[w] for w in ex['question']])
#     question_char = torch.LongTensor([char_dict[w] for w in ex['question_char']])
#
#     # Create extra features vector
#     if len(feature_dict) > 0:
#         c_features = torch.zeros(len(ex['document']), len(feature_dict))
#         q_features = torch.zeros(len(ex['question']), len(feature_dict))
#     else:
#         c_features = None
#         q_features = None
#
#     # f_{exact_match}
#     if args.use_exact_match:
#         q_words_cased = {w for w in ex['question']}
#         q_words_uncased = {w.lower() for w in ex['question']}
#         q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
#         for i in range(len(ex['document'])):
#             if ex['document'][i] in q_words_cased:
#                 c_features[i][feature_dict['in_cased']] = 1.0
#             if ex['document'][i].lower() in q_words_uncased:
#                 c_features[i][feature_dict['in_uncased']] = 1.0
#             if q_lemma and ex['clemma'][i] in q_lemma:
#                 c_features[i][feature_dict['in_lemma']] = 1.0
#
#         c_words_cased = {w for w in ex['document']}
#         c_words_uncased = {w.lower() for w in ex['document']}
#         c_lemma = {w for w in ex['clemma']} if args.use_lemma else None
#         for i in range(len(ex['question'])):
#             if ex['question'][i] in c_words_cased:
#                 q_features[i][feature_dict['in_cased']] = 1.0
#             if ex['question'][i].lower() in c_words_uncased:
#                 q_features[i][feature_dict['in_uncased']] = 1.0
#             if c_lemma and ex['qlemma'][i] in c_lemma:
#                 q_features[i][feature_dict['in_lemma']] = 1.0
#
#     # f_{token} (POS)
#     if args.use_pos:
#         for i, w in enumerate(ex['cpos']):
#             f = 'pos=%s' % w
#             if f in feature_dict:
#                 c_features[i][feature_dict[f]] = 1.0
#         for i, w in enumerate(ex['qpos']):
#             f = 'pos=%s' % w
#             if f in feature_dict:
#                 q_features[i][feature_dict[f]] = 1.0
#
#     # f_{token} (NER)
#     if args.use_ner:
#         for i, w in enumerate(ex['cner']):
#             f = 'ner=%s' % w
#             if f in feature_dict:
#                 c_features[i][feature_dict[f]] = 1.0
#         for i, w in enumerate(ex['qner']):
#             f = 'ner=%s' % w
#             if f in feature_dict:
#                 q_features[i][feature_dict[f]] = 1.0
#
#     # f_{token} (TF)
#     if args.use_tf:
#         counter = Counter([w.lower() for w in ex['document']])
#         l = len(ex['document'])
#         for i, w in enumerate(ex['document']):
#             c_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
#         counter = Counter([w.lower() for w in ex['question']])
#         l = len(ex['question'])
#         for i, w in enumerate(ex['question']):
#             q_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
#
#     # Maybe return without target
#     if 'answers' not in ex:
#         return document, document_char, c_features, question, question_char, q_features, ex['id']
#
#     # ...or with target(s) (might still be empty if answers is empty)
#     if single_answer:
#         assert(len(ex['answers']) > 0)
#         start = torch.LongTensor(1).fill_(ex['answers'][0][0])
#         end = torch.LongTensor(1).fill_(ex['answers'][0][1])
#     else:
#         start = [a[0] for a in ex['answers']]
#         end = [a[1] for a in ex['answers']]
#
#     selected_offset = None
#
#     if args.use_sentence_selector:
#         sentence_boundaries = []
#         counter = 0
#         for sent in ex['sentences']:
#             sentence_boundaries.append([counter, counter + len(sent)])
#             counter += len(sent)
#         return sentence_boundaries, ex["offsets"], start, end, sent_selector_vectorize(ex, model, single_answer)
#
#     return document, document_char, c_features, question, question_char, q_features, start, end, ex['id']

def vectorize(ex, model, single_answer=False):
    """Torchify a single example."""
    args = model.args
    word_dict = model.word_dict
    char_dict = model.char_dict
    feature_dict = model.feature_dict
    ex['pos'] = ex['cpos']
    ex['ner'] = ex['cner']
    ex['lemma'] = ex['clemma']

    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    document_char = torch.LongTensor([char_dict[w] for w in ex['document_char']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    question_char = torch.LongTensor([char_dict[w] for w in ex['question_char']])


    sentence_boundaries = []
    counter = 0
    for sent in ex['sentences']:
        sentence_boundaries.append([counter, counter + len(sent)])
        counter += len(sent)

    # If Sentence Selector is turned on, then run the document through and get the top sentence
    selected_offset = None
    if args.use_sentence_selector:
        # sentence_lengths = [len(sent) for sent in ex['sentences']]
        # max_length = max(sentence_lengths)
        # sentences = torch.LongTensor([pad_single_seq([word_dict[w] for w in sent], max_length) for sent in ex['sentences']])

        if args.use_gold_sentence:
            # Use gold sentence
            if len(ex['gold_sentence_ids']) == 0:
                return []
            top_sentence = [ex['gold_sentence_ids'][0]]
        else:
            ex_batch = sent_selector_batchify([sent_selector_vectorize(ex, model.sentence_selector, single_answer)])

            if args.dynamic_selector:
                top_sentence = model.sentence_selector.predict(ex_batch, use_threshold = args.selection_threshold)[0]
            else:
                top_sentence = model.sentence_selector.predict(ex_batch)[0]
            #if len(ex['gold_sentence_ids']) > 0 and top_sentence not in ex['gold_sentence_ids']:
            #    return []
        # Extract top sentence and change ex["document"] accordingly
        tokens = []
        pos = []
        ner = []
        lemma = []
        offset_subset = []
        assert len(ex['pos']) == len(ex['ner']) == len(ex['lemma'])
        ## also need to add characters
        for t in top_sentence:
            tokens += ex['sentences'][t]
            window = sentence_boundaries[t]
            pos += ex['pos'][window[0]:window[1]]
            ner += ex['ner'][window[0]:window[1]]
            lemma += ex['lemma'][window[0]:window[1]]
            offset_subset += ex["offsets"][sentence_boundaries[t][0]:sentence_boundaries[t][1]]

        # Reconstruct based on what you are selecting
        document = torch.LongTensor([word_dict[w] for w in tokens])
        ex['document'] = tokens
        ex['cpos'] = pos
        ex['cner'] = ner
        ex['clemma'] = lemma
        selected_offset = offset_subset
        ## Get first letters of the current document tokens
        document_char = torch.LongTensor([tokens[i][0] if len(tokens[i]) > 0 else '' for i in range(len(tokens))])


        # Check if selected sentence contains any answer span
        # account for answers being in between the gold sentence
        flag = True
        window = sentence_boundaries[top_sentence[0]]
        for answer in ex['answers']:
            if answer[0] >= window[0] and answer[1] < window[1]:
                new_start = answer[0] - window[0]
                new_end = answer[1] - window[0]
                flag = False
                break
            elif answer[0] >= window[0] and answer[1] < sentence_boundaries[top_sentence[0] + 1][1]:
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
        c_features = torch.zeros(len(ex['document']), len(feature_dict))
        q_features = torch.zeros(len(ex['question']), len(feature_dict))
    else:
        c_features = None
        q_features = None

    # f_{exact_match}
    if args.use_exact_match:
        q_words_cased = {w for w in ex['question']}
        q_words_uncased = {w.lower() for w in ex['question']}
        q_lemma = {w for w in ex['qlemma']} if args.use_lemma else None
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                c_features[i][feature_dict['in_cased']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                c_features[i][feature_dict['in_uncased']] = 1.0
            if q_lemma and ex['clemma'][i] in q_lemma:
                c_features[i][feature_dict['in_lemma']] = 1.0

        c_words_cased = {w for w in ex['document']}
        c_words_uncased = {w.lower() for w in ex['document']}
        c_lemma = {w for w in ex['clemma']} if args.use_lemma else None
        for i in range(len(ex['question'])):
            if ex['question'][i] in c_words_cased:
                q_features[i][feature_dict['in_cased']] = 1.0
            if ex['question'][i].lower() in c_words_uncased:
                q_features[i][feature_dict['in_uncased']] = 1.0
            if c_lemma and ex['qlemma'][i] in c_lemma:
                q_features[i][feature_dict['in_lemma']] = 1.0

    # f_{token} (POS)
    if args.use_pos:
        for i, w in enumerate(ex['cpos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0
        for i, w in enumerate(ex['qpos']):
            f = 'pos=%s' % w
            if f in feature_dict:
                q_features[i][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, w in enumerate(ex['cner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                c_features[i][feature_dict[f]] = 1.0
        for i, w in enumerate(ex['qner']):
            f = 'ner=%s' % w
            if f in feature_dict:
                q_features[i][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in ex['document']])
        l = len(ex['document'])
        for i, w in enumerate(ex['document']):
            c_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l
        counter = Counter([w.lower() for w in ex['question']])
        l = len(ex['question'])
        for i, w in enumerate(ex['question']):
            q_features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    # Maybe return without target
    if 'answers' not in ex:
        return document, document_char, c_features, question, question_char, q_features, ex['id']

    # ...or with target(s) (might still be empty if answers is empty)
    # put sentence selection criterion here
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

    return document, document_char, c_features, question, question_char, q_features, selected_offset, start, end, ex['id']


def batchify(batch):
    """Gather a batch of individual examples into one batch."""
    NUM_INPUTS = 6
    NUM_TARGETS = 2
    NUM_EXTRA = 2

    docs = [ex[0] for ex in batch]
    doc_chars = [ex[1] for ex in batch]
    c_features = [ex[2] for ex in batch]
    questions = [ex[3] for ex in batch]
    question_chars = [ex[4] for ex in batch]
    q_features = [ex[5] for ex in batch]
    offsets = [ex[6] for ex in batch]
    ids = [ex[-1] for ex in batch]

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    x1 = torch.LongTensor(len(docs), max_length).zero_()
    x1_c = torch.LongTensor(len(docs), max_length).zero_()
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if c_features[0] is None:
        x1_f = None
    else:
        x1_f = torch.zeros(len(docs), max_length, c_features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:
            x1_f[i, :d.size(0)].copy_(c_features[i])
    for i, c in enumerate(doc_chars):
        x1_c[i, :c.size(0)].copy_(c)

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).zero_()
    x2_c = torch.LongTensor(len(questions), max_length).zero_()
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    if q_features[0] is None:
        x2_f = None
    else:
        x2_f = torch.zeros(len(questions), max_length, q_features[0].size(1))
    for i, d in enumerate(questions):
        x2[i, :d.size(0)].copy_(d)
        x2_mask[i, :d.size(0)].fill_(0)
        if x2_f is not None:
            x2_f[i, :d.size(0)].copy_(q_features[i])
    for i, c in enumerate(question_chars):
        x2_c[i, :c.size(0)].copy_(c)

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, ids

    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        # ...Otherwise add targets
        if torch.is_tensor(batch[0][NUM_INPUTS]):
            y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
            y_e = torch.cat([ex[NUM_INPUTS+1] for ex in batch])
        else:
            y_s = [ex[NUM_INPUTS] for ex in batch]
            y_e = [ex[NUM_INPUTS+1] for ex in batch]
    else:
        raise RuntimeError('Incorrect number of inputs per example.')

    return x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, y_s, y_e, offsets, ids
