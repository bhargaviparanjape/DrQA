
import atexit
import errno
import gzip
import json
import numpy as np
import os
import pickle
import requests
import socket
import sys
import subprocess
import time
import argparse

DEVNULL = open(os.devnull, 'w')

def run_model(json_filename, out_dir, args, verbose = False):
    if verbose:
        pipeout = None
    else:
        pipeout = DEVNULL
    env = os.environ.copy()
    env['PYTHONPATH'] = args.drqa_path
    run_args = [
        'python3', 'scripts/reader/predict_expected_scores.py', '--data']

    subprocess.check_call(run_args, env=env, stdout=pipeout, stderr=pipeout)
    output_path = os.path.join(out_dir, "eval.json")
    with open(output_path) as f:
        data = json.load(f)
        scores = data["scores"]
        preds = data["preds"]
    return scores, preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drqa-path', type=str, default=None) # ROOT_DIR
    parser.add_argument('--dev-file', type=str, default=None) # json file
    parser.add_argument('--embed-dir', type=str, default=None) # GLOVE-DIR
    parser.add_argument('--pretrained-model', type=str, default=None) # path to .mdl file (change to oracle if using sentence selector)

    parser.add_argument('--use-sentence-selector', action="store_true", default=False)
    parser.add_argument('--sentence-selector', type=str, default=None) # path to selector .mdl file

    args = parser.parse_args()

    run_model(args.dev_file, args, None)
