#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Preprocess the SQuAD dataset for training."""

import argparse
import os
import sys
import json
import time

from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
from drqa import tokenizers

# ------------------------------------------------------------------------------
# Tokenize + annotate.
# ------------------------------------------------------------------------------

TOK = None


def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)


def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
        'sentence_boundaries' : tokens.sentences(),
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------


def load_dataset(path):
    """Load json file and store fields separately."""
    with open(path) as f:
        data = json.load(f)['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for idx, article in enumerate(data):
        if idx > 2 and args.truncate:
            break
        for i, paragraph in enumerate(article['paragraphs']):
            if i > 10:
                break
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output


def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]


def process_dataset(data, tokenizer, workers=None):
    """Iterate processing (tokenize, parse, etc) dataset multithreaded."""
    tokenizer_class = tokenizers.get_class(tokenizer)
    make_pool = partial(Pool, workers, initializer=init)

    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}, 'classpath' : "/home/bhargavi/robust_nlp/invariance/DrQA/data/corenlp/*"}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'},'classpath' : "/home/bhargavi/robust_nlp/invariance/DrQA/data/corenlp/*"})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    ## code to override Pool
    # init(tokenizer_class, {'annotators': {'lemma'}, 'classpath' : "/home/bhargavi/robust_nlp/invariance/DrQA/data/corenlp/*"})
    # q_tokens = []
    # for idx in range(len(data['questions'])):
    #     q_tokens.append(tokenize(data['questions'][idx]))
    # c_tokens = []
    # for idx in range(len(data['contexts'])):
    #     c_tokens.append(tokenize(data['contexts'][idx]))

    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        context_sentence_boundaries = c_tokens[data['qid2cid'][idx]]['sentence_boundaries']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)
        ## sentences
        ans_tokens_list = list(set(ans_tokens))
        sentences = []
        gold_sentence_ids = []
        for idx, tup in enumerate(context_sentence_boundaries):
            for a in ans_tokens_list:
                if a[0] >= tup[0] and a[1] < tup[1]:
                    gold_sentence_ids.append(idx)
                elif a[0] >= tup[0] and a[0] < tup[1] and a[1] >= tup[1]:
                    gold_sentence_ids.append(idx)
                    gold_sentence_ids.append(idx+1)
            sentence = document[tup[0]:tup[1]]
            sentences.append(sentence)
        gold_sentence_ids_set = list(set(gold_sentence_ids))
        if len(ans_tokens_list) == 0:
            print("No golden sentence available")
        ## gold_sentence_id
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
            'sentences': sentences,
            'gold_sentence_ids' : gold_sentence_ids_set,
        }


# -----------------------------------------------------------------------------
# Commandline options
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory')
    parser.add_argument('--out_dir', type=str, help='Path to output file dir')
    parser.add_argument('--split', type=str, help='Filename for train/dev split',
                        default='SQuAD-v1.1-train')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--tokenizer', type=str, default='corenlp')
    parser.add_argument('--truncate', action="store_true", default=False)
    args = parser.parse_args()

    t0 = time.time()

    in_file = os.path.join(args.data_dir, args.split + '.json')
    print('Loading dataset %s' % in_file, file=sys.stderr)
    dataset = load_dataset(in_file)

    out_file = os.path.join(
        args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
    )
    print('Will write to file %s' % out_file, file=sys.stderr)
    with open(out_file, 'w') as f:
        for ex in process_dataset(dataset, args.tokenizer, args.workers):
            f.write(json.dumps(ex) + '\n')
    print('Total time: %.4f (s)' % (time.time() - t0))
