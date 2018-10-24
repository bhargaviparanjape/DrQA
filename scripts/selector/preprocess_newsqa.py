import argparse
import os
import sys
import json
import time
import csv
import pdb

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
		answer_ranges = [tuple([int(range_.split(':')[0]), int(range_.split(':')[1])]) for range_ in data[4].split(',')]
		# answer_range = tuple([int(el) for el in data[4].split(":")])
		answers = [[document[a] for a in range(ans[0], ans[1])] for ans in answer_ranges]
		start = [a[0] for a in answer_ranges]
		end = [a[1] for a in answer_ranges]
		sentence_starts = [0] + [int(a) for a in data[5].split(',')]
		sentences = [document[sentence_starts[i]:sentence_starts[i+1]] for i in range(len(sentence_starts)-1)]
		gold_sentence_ids = []
		for pos in range(len(sentence_starts)-1):
			for tup in zip(start, end):
				s = tup[0]
				e = tup[1]
				if s >= sentence_starts[pos] and e < sentence_starts[pos+1]:
					gold_sentence_ids.append(pos)
				elif s >= sentence_starts[pos] and e >= sentence_starts[pos+1]:
					gold_sentence_ids.append(pos)
					gold_sentence_ids.append(pos+1)
		if len(gold_sentence_ids)  == 0:
			print("No golden sentence available")
		yield {
			'id': question_id,
			'story_id': context_id,
			'question': question,
			'document': document,
			'answers': answers,
			'start_token': start,
			'end_token': end,
			'sentences' : sentences,
			'gold_sentence_ids' : gold_sentence_ids
		}


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to SQuAD data directory')
parser.add_argument('--out_dir', type=str, help='Path to output file dir')
parser.add_argument('--split', type=str, help='Filename for train/dev split',
                    default='NewsQA-v1.0-train')

args = parser.parse_args()

t0 = time.time()

in_file = os.path.join(args.data_dir, args.split + '.csv')
print('Loading dataset %s' % in_file, file=sys.stderr)
dataset = load_dataset(in_file)


out_file = os.path.join(
    args.out_dir, '%s-processed-%s.txt' % (args.split, "corenlp")
)

with open(out_file, 'w') as f:
	for ex in process_dataset(dataset):
		f.write(json.dumps(ex) + '\n')

print('Total time: %.4f (s)' % (time.time() - t0))
