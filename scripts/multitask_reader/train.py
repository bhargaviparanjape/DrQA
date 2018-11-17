#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main DrQA reader training script."""

import argparse
import json
import logging
import os
import subprocess
import sys
from os.path import dirname, realpath
import pdb
import numpy as np
import torch

sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

# For GloVe based reader
from drqa.multitask_reader import utils, config
from drqa.multitask_reader import data as reader_data, vector as reader_vector
from drqa.selector import data as selector_data, vector as selector_vector
from drqa.multitask_reader import DocReader

'''
# from drqa.elmo_reader import utils, config
# from drqa.elmo_reader import data as reader_data, vector as reader_vector
# from drqa.selector import data as selector_data, vector as selector_vector
# from drqa.elmo_reader import DocReader
'''

from drqa.selector import SentenceSelector
from scripts.selector.train import validate_unofficial as validate_selector
from drqa import DATA_DIR as DRQA_DATA

logger = logging.getLogger()


# ------------------------------------------------------------------------------
# Training arguments.
# ------------------------------------------------------------------------------


# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = '/tmp/drqa-models/'
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def set_defaults(args):
    """Make sure the commandline arguments are initialized properly."""
    # Check critical files exist
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError('No such file: %s' % args.dev_json)
    train_files = []
    for t_file in args.train_file:
        fullpath = os.path.join(args.data_dir, t_file)
        train_files.append(fullpath)
    vars(args)["train_file"] = train_files
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)

    # Adversarial files

    if args.adv_dev_file is not None and args.adv_dev_json is not None:
        adv_dev_files = []
        for t_file in args.adv_dev_file:
            fullpath = os.path.join(args.data_dir, t_file)
            adv_dev_files.append(fullpath)
        vars(args)["adv_dev_file"] = adv_dev_files
        adv_dev_json = []
        for t_file in args.adv_dev_json:
            fullpath = os.path.join(args.data_dir, t_file)
            adv_dev_json.append(fullpath)
        vars(args)["adv_dev_json"] = adv_dev_json

    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)
    # Set model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Set model name
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Set log + model file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Embeddings options
    if args.embedding_file:
        with open(args.embedding_file) as f:
            dim = len(f.readline().strip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Make sure tune_partial and fix_embeddings are consistent.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Make sure fix_embeddings and embedding_file are consistent
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', str2bool)

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=1,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')
    runtime.add_argument('--global_mode', type=str, default="train", help="global mode: {train, test}")
    runtime.add_argument("--patience", type=int, default=10, help="how many bad iterations for early stopping")

    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str, action = "append",
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-corenlp.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.6B.300d.txt',
                       help='Space-separated pretrained embeddings file')
    files.add_argument('--adv-dev-file', type=str,
                       action="append", default=[],
                       help='Preprocessed dev file')
    files.add_argument('--adv-dev-json', type=str, action="append", default=[],
                       help=('Unprocessed dev file to run validation '
                             'while training on'))

    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Data preprocessing
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


# ------------------------------------------------------------------------------
# Initalization from scratch.
# ------------------------------------------------------------------------------


def init_from_scratch(args, train_exs, dev_exs):
    """New model, new data, new dictionary."""
    # Create a feature dict out of the annotations in the data
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Build a dictionary from the data questions + words (train/dev splits)
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.build_word_dict(args, train_exs + dev_exs)
    logger.info('Num words = %d' % len(word_dict))

    # Initialize model
    # model = DocReader(config.get_model_args(args), word_dict, feature_dict)
    # Send all arguments
    model = DocReader(args, word_dict, feature_dict)

    # Load pretrained embeddings for words in dictionary
    if args.embedding_file:
        model.load_embeddings(word_dict.tokens(), args.embedding_file)

    return model


# ------------------------------------------------------------------------------
# Train loop.
# ------------------------------------------------------------------------------


def train(args, data_loader, model, global_stats):
    """Run through one epoch of model training with the provided data loader."""
    # Initialize meters + timers
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run one epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


# ------------------------------------------------------------------------------
# Validation loops. Includes both "unofficial" and "official" functions that
# use different metrics and implementations.
# ------------------------------------------------------------------------------


def validate_unofficial(args, data_loader, model, global_stats, mode):
    """Run one full unofficial validation.
    Unofficial = doesn't use SQuAD script.
    """
    eval_time = utils.Timer()
    start_acc = utils.AverageMeter()
    end_acc = utils.AverageMeter()
    exact_match = utils.AverageMeter()
    support_acc = utils.AverageMeter()

    # Make predictions
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s, pred_e, _, pred_sp = model.predict(ex)
        target_s, target_e, target_sp = ex[-5:-2]

        # We get metrics for independent start/end and joint start/end
        accuracies = eval_accuracies_spans(pred_s, target_s, pred_e, target_e)
        support_accuracy = eval_accuracies_support(pred_sp, target_sp, mode)
        start_acc.update(accuracies[0], batch_size)
        end_acc.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)
        support_acc.update(support_accuracy, batch_size)

        # If getting train accuracies, sample max 10k
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial: Epoch = %d | start = %.2f | ' %
                (mode, global_stats['epoch'], start_acc.avg) +
                'end = %.2f | exact = %.2f | examples = %d | ' %
                (end_acc.avg, exact_match.avg, examples) +
                'sentence_selection = %.2f | ' % (support_acc.avg) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'exact_match': exact_match.avg}


def validate_official(args, data_loader, model, global_stats,
                      offsets, texts, answers):
    """Run one full official validation. Uses exact spans and same
    exact match/F1 score computation as in the SQuAD script.

    Extra arguments:
        offsets: The character start/end indices for the tokens in each context.
        texts: Map of qid --> raw text of examples context (matches offsets).
        answers: Map of qid --> list of accepted answers.
    """
    clean_id_file = open(os.path.join(DATA_DIR, "clean_qids.txt"), "w+")
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    # Run through examples
    examples = 0
    bad_examples = 0
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        chosen_offset = ex[-2]
        pred_s, pred_e, _ , pred_sp = model.predict(ex)

        for i in range(batch_size):
            if pred_s[i][0] >= len(offsets[ex_id[i]]) or pred_e[i][0] >= len(offsets[ex_id[i]]):
                bad_examples += 1
                continue
            if args.use_sentence_selector:
                s_offset = chosen_offset[i][pred_s[i][0]][0]
                e_offset = chosen_offset[i][pred_e[i][0]][1]
            else:
                s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
                e_offset = offsets[ex_id[i]][pred_e[i][0]][1]

            # If sentence selector is not turned on
            if not args.use_sentence_selector or args.select_k == 1:
                prediction = texts[ex_id[i]][s_offset:e_offset]

            if args.select_k > 1:
                prediction = ""
                offset_subset = chosen_offset[i][pred_s[i][0]: pred_e[i][0] + 1]
                for enum_, o in enumerate(offset_subset):
                    prediction += texts[ex_id[i]][o[0]:o[1]] + " "
                prediction = prediction.strip()

            # Compute metrics
            ground_truths = answers[ex_id[i]]
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))

            f1_example = utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths)

            if f1_example != 0:
                clean_id_file.write(ex_id[i] + "\n")


        examples += batch_size

    clean_id_file.close()
    logger.info('dev valid official: Epoch = %d | EM = %.2f | ' %
                (global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))
    logger.info('Bad Offset Examples during official eval: %d' % bad_examples)
    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies_spans(pred_s, target_s, pred_e, target_e):
    """An unofficial evaluation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    # Convert 1D tensors to lists of lists (compatibility)
    if torch.is_tensor(target_s):
        target_s = [[e.data.numpy()] for e in target_s]
        target_e = [[e.data.numpy()] for e in target_e]


    # Compute accuracies from targets
    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        # Start matches
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        # End matches
        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        # Both start and end match
        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100


def eval_accuracies_support(pred, target, mode="dev"):
    if torch.is_tensor(target):
        target = [[e] for e in target]
    elif torch.is_tensor(target[0]):
        target = [[e.item()] for e in target[0]]
    else:
        target = target # not in a tuple anymore
    batch_size = len(pred)
    accuracy = utils.AverageMeter()
    for i in range(batch_size):
        flag = False
        for j in pred[i]:
            if j in target[i]:
                flag = True
                break
        if flag:
            accuracy.update(1)
        else:
            accuracy.update(0)
    return accuracy.avg * 100

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


def validate_adversarial(args, model, global_stats, mode="dev"):
    # create dataloader for dev sets, load thier jsons, integrate the function


    for idx, dataset_file in enumerate(args.adv_dev_json):

        predictions = {}

        logger.info("Validating Adversarial Dataset %s" % dataset_file)
        exs = utils.load_data(args, args.adv_dev_file[idx])
        logger.info('Num dev examples = %d' % len(exs))
        ## Create dataloader
        dev_dataset = reader_data.ReaderDataset(exs, model, single_answer=False)
        if args.sort_by_len:
            dev_sampler = reader_data.SortedBatchSampler(dev_dataset.lengths(),
                                                  args.test_batch_size,
                                                  shuffle=False)
        else:
            dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        if args.use_sentence_selector:
            dev_batcher = reader_vector.sentence_batchifier(model, single_answer=False)
            #batching_function = dev_batcher.batchify
            batching_function = reader_vector.batchify
        else:
            batching_function = reader_vector.batchify
        dev_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batching_function,
            pin_memory=args.cuda,
        )

        texts = utils.load_text(dataset_file)
        offsets = {ex['id']: ex['offsets'] for ex in exs}
        answers = utils.load_answers(dataset_file)

        eval_time = utils.Timer()
        f1 = utils.AverageMeter()
        exact_match = utils.AverageMeter()

        examples = 0
        bad_examples = 0
        for ex in dev_loader:
            ex_id, batch_size = ex[-1], ex[0].size(0)
            chosen_offset = ex[-2]
            pred_s, pred_e, _, pred_sp = model.predict(ex)

            for i in range(batch_size):
                if pred_s[i][0] >= len(offsets[ex_id[i]]) or pred_e[i][0] >= len(offsets[ex_id[i]]):
                    bad_examples += 1
                    continue
                if args.use_sentence_selector:
                    s_offset = chosen_offset[i][pred_s[i][0]][0]
                    e_offset = chosen_offset[i][pred_e[i][0]][1]
                else:
                    s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
                    e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
                prediction = texts[ex_id[i]][s_offset:e_offset]

                if args.select_k > 1:
                    prediction = ""
                    offset_subset = chosen_offset[i][pred_s[i][0]: pred_e[i][0]]
                    for enum_, o in enumerate(offset_subset):
                        prediction += texts[ex_id[i]][o[0]:o[1]] + " "
                    prediction = prediction.strip()

                predictions[ex_id[i]] = prediction

                ground_truths = answers[ex_id[i]]
                exact_match.update(utils.metric_max_over_ground_truths(
                    utils.exact_match_score, prediction, ground_truths))
                f1.update(utils.metric_max_over_ground_truths(
                    utils.f1_score, prediction, ground_truths))

            examples += batch_size

        logger.info('dev valid official for dev file %s : Epoch = %d | EM = %.2f | ' %
                    (dataset_file, global_stats['epoch'], exact_match.avg * 100) +
                    'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                    (f1.avg * 100, examples, eval_time.time()))

        orig_f1_score = 0.0
        orig_exact_match_score = 0.0
        adv_f1_scores = {}  # Map from original ID to F1 score
        adv_exact_match_scores = {}  # Map from original ID to exact match score
        adv_ids = {}
        all_ids = set()  # Set of all original IDs
        f1 = exact_match = 0
        dataset = json.load(open(dataset_file))['data']
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    orig_id = qa['id'].split('-')[0]
                    all_ids.add(orig_id)
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + ' will receive score 0.'
                        # logger.info(message)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    cur_exact_match = utils.metric_max_over_ground_truths(utils.exact_match_score,
                                                                    prediction, ground_truths)
                    cur_f1 = utils.metric_max_over_ground_truths(utils.f1_score, prediction, ground_truths)
                    if orig_id == qa['id']:
                        # This is an original example
                        orig_f1_score += cur_f1
                        orig_exact_match_score += cur_exact_match
                        if orig_id not in adv_f1_scores:
                            # Haven't seen adversarial example yet, so use original for adversary
                            adv_ids[orig_id] = orig_id
                            adv_f1_scores[orig_id] = cur_f1
                            adv_exact_match_scores[orig_id] = cur_exact_match
                    else:
                        # This is an adversarial example
                        if (orig_id not in adv_f1_scores or adv_ids[orig_id] == orig_id
                            or adv_f1_scores[orig_id] > cur_f1):
                            # Always override if currently adversary currently using orig_id
                            adv_ids[orig_id] = qa['id']
                            adv_f1_scores[orig_id] = cur_f1
                            adv_exact_match_scores[orig_id] = cur_exact_match
        orig_f1 = 100.0 * orig_f1_score / len(all_ids)
        orig_exact_match = 100.0 * orig_exact_match_score / len(all_ids)
        adv_exact_match = 100.0 * sum(adv_exact_match_scores.values()) / len(all_ids)
        adv_f1 = 100.0 * sum(adv_f1_scores.values()) / len(all_ids)
        logger.info("For the file %s Original Exact Match : %.4f ; Original F1 : : %.4f | "
                    % (dataset_file, orig_exact_match, orig_f1)
                    + "Adversarial Exact Match : %.4f ; Adversarial F1 : : %.4f " % (adv_exact_match, adv_f1))



def main(args):
    
    # --------------------------------------------------------------------------
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = []  
    for t_file in args.train_file:
        train_exs += utils.load_data(args, t_file, skip_no_answer=True)
    np.random.shuffle(train_exs)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))
    
    # If we are doing offician evals then we need to:
    # 1) Load the original text to retrieve spans from offsets.
    # 2) Load the (multiple) text answers for each question.
    if args.official_eval:
        dev_texts = utils.load_text(args.dev_json)
        dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
        dev_answers = utils.load_answers(args.dev_json)


    ## OFFSET comes from the gold sentence; the predicted sentence value shoule be maintained and sent to official validation set
    # --------------------------------------------------------------------------
    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            if args.expand_dictionary:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.load_words(args, train_exs + dev_exs)
                added = model.expand_dictionary(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.load_embeddings(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # --------------------------------------------------------------------------
    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    # Sentence selection objective : run the sentence selector as a submodule
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = reader_data.ReaderDataset(train_exs, model, single_answer=True)
    # Filter out None examples in training dataset (where sentence selection fails)

    #train_dataset.examples = [t for t in train_dataset.examples if t is not None]
    if args.sort_by_len:
        train_sampler = reader_data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    if args.use_sentence_selector:
        train_batcher = reader_vector.sentence_batchifier(model, single_answer=True)
        # batching_function = train_batcher.batchify
        batching_function = reader_vector.batchify
    else:
        batching_function = reader_vector.batchify
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=batching_function,
        pin_memory=args.cuda,
    )
    dev_dataset = reader_data.ReaderDataset(dev_exs, model, single_answer=False)
    #dev_dataset.examples = [t for t in dev_dataset.examples if t is not None]
    if args.sort_by_len:
        dev_sampler = reader_data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)

    if args.use_sentence_selector:
        dev_batcher = reader_vector.sentence_batchifier(model, single_answer=False)
        # batching_function = dev_batcher.batchify
        batching_function = reader_vector.batchify
    else:
        batching_function = reader_vector.batchify
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=batching_function,
        pin_memory=args.cuda,
    )
    
    ## Dev dataset for measuring performance of the trained sentence selector
    if args.use_sentence_selector:
        dev_dataset1 = selector_data.SentenceSelectorDataset(dev_exs, model.sentence_selector, single_answer=False)
        #dev_dataset1.examples = [t for t in dev_dataset.examples if t is not None]
        if args.sort_by_len:
            dev_sampler1 = selector_data.SortedBatchSampler(dev_dataset1.lengths(),
                                                  args.test_batch_size,
                                                  shuffle=False)
        else:
            dev_sampler1 = torch.utils.data.sampler.SequentialSampler(dev_dataset1)
        dev_loader1 = torch.utils.data.DataLoader(
            dev_dataset1,
            #batch_size=args.test_batch_size,
            #sampler=dev_sampler1,
            batch_sampler = dev_sampler1,
            num_workers=args.data_workers,
            collate_fn=selector_vector.batchify,
            pin_memory=args.cuda,
        )


    # -------------------------------------------------------------------------
    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))
    

    # --------------------------------------------------------------------------
    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}

    # --------------------------------------------------------------------------
    # QUICKLY VALIDATE ON PRETRAINED MODEL
    
    if args.global_mode == "test":
        result1 = validate_unofficial(args, dev_loader, model, stats, mode='dev')
        result2 = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)
        print(result2[args.valid_metric])
        print(result1["exact_match"])
        if args.use_sentence_selector:
            sent_stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
            #sent_selector_results = validate_selector(model.sentence_selector.args, dev_loader1, model.sentence_selector, sent_stats, mode="dev")
            #print("Sentence Selector model acheives:")
            #print(sent_selector_results["accuracy"])

        if len(args.adv_dev_json) > 0:
            validate_adversarial(args, model, stats, mode="dev")
        exit(0)



    valid_history = []
    bad_counter = 0 
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate unofficial (train)
        validate_unofficial(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = validate_unofficial(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = validate_official(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        # Save best valid
        if result[args.valid_metric] >= stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter > args.patience:
            logger.info("Early Stopping at epoch: %d" % epoch)
            exit(0)


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_args(parser)
    args = parser.parse_args()
    set_defaults(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
