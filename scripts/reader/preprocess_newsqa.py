import argparse
import os
import sys
import json
import time
import csv

def load_dataset(path):
	output = []
	with open(path) as f:
		csv_reader = csv.reader(f)
		header = next(csv_reader)
		data = [row for row in csv_reader]
		for idx, row in enumerate(data):
			output.append(["qid" + str(idx)] + list(row))
	return output

def process_dataset(dataset):
	for idx, data in enumerate(dataset):
		question_id = data[0]
		context_id = data[1]
		document = [s.strip() for s in data[2].split()]
		question = [s.strip() for s in data[3].split()]
		answer_range = tuple([int(el) for el in data[4].split(":")])
		answers = [document[a] for a in range(answer_range)]
		start = answer_range[0]
		end = answer_range[1]
		yield {
			'id' : question_id,
			'story_id' : context_id,
			'question': question,
			'document': document,
			'answers' : answers,
			'start_token' : start,
			'end_token' : end
		}


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='NewsQA-v1.0-train')

args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.json')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)


out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, args.tokenizer)
)

with open(out_file, 'w') as f:
    for ex in process_dataset(dataset, args.tokenizer, args.workers):
        f.write(json.dumps(ex) + '\n')

print('Total time: %.4f (s)' % (time.time() - t0))

