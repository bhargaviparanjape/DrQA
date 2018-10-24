import argparse
import json
import logging
import os
import subprocess
import sys
from os.path import dirname, realpath

import numpy as np
import torch

sys.path.append(dirname(dirname(dirname(realpath(__file__)))))
from drqa.reader import utils, config
from drqa.reader import data as reader_data, vector as reader_vector

from drqa.reader import DocReader
from drqa import DATA_DIR as DRQA_DATA

logger = logging.getLogger()

# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = '/tmp/drqa-models/'
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')
BEAM_SIZE = 10

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
                       action="append",
                       help='Preprocessed dev file')
    files.add_argument('--adv-dev-json', type=str, action="append",
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

def get_y_pred_beam(start_probs, end_probs, beam_size=BEAM_SIZE):
  beam = []
  for i, p_start in enumerate(start_probs):
    for j, p_end in enumerate(end_probs):
      if i <= j:
        beam.append((i, j + 1, p_start * p_end))
  beam.sort(key=lambda x: x[2], reverse=True)
  return beam[:beam_size]

def compute_expected_metric(args, data_loader, model, global_stats,
                      offsets, texts, answers):
    scores = {}
    preds = {}
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        chosen_offset = ex[-2]
        tup, score_s, score_e= model.predict_probs(ex)
        pred_s, pred_e , _ = tup
        for i in range(batch_size):
            if args.use_sentence_selector:
                s_offset = chosen_offset[i][pred_s[i][0]][0]
                e_offset = chosen_offset[i][pred_e[i][0]][1]
            else:
                s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
                e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
            prediction = texts[ex_id[i]][s_offset:e_offset]
            ground_truths = answers[ex_id[i]]

            beam = get_y_pred_beam(score_s[i].numpy(), score_e[i].numpy(), BEAM_SIZE)
            total_prob = sum(x[2] for x in beam)
            score = 0.0
            for (start, end, prob) in beam:
                if args.use_sentence_selector:
                    s_offset = chosen_offset[i][start][0]
                    e_offset = chosen_offset[i][end][1]
                else:
                    s_offset = offsets[ex_id[i]][start][0]
                    e_offset = offsets[ex_id[i]][end][1]
                phrase = texts[ex_id[i]][s_offset:e_offset]
                cur_f1 = utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths)
                score += prob / total_prob * cur_f1
            scores[ex_id[i]] = score
            preds[ex_id[i]] = prediction
    return scores, preds



def main(args):
    dev_exs = utils.load_data(args, args.dev_file)
    dev_texts = utils.load_text(args.dev_json)
    dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
    dev_answers = utils.load_answers(args.dev_json)
    model = DocReader.load(args.pretrained, args)
    model.init_optimizer()
    if args.cuda:
        model.cuda()
    if args.parallel:
        model.parallelize()
    dev_dataset = reader_data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_sampler = reader_data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=reader_vector.batchify,
        pin_memory=args.cuda,
    )
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    scores, pred_objs = compute_expected_metric(args, dev_loader, model, stats,
                      dev_offsets, dev_texts, dev_answers)
    return scores, pred_objs

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
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    main(args)
