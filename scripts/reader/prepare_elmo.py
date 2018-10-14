from allennlp.modules.elmo import Elmo, batch_to_ids
import os,sys,argparse,numpy as np,pdb,json
from drqa.reader import utils, config
from drqa import tokenizers
from multiprocessing import Pool
from multiprocessing.util import Finalize
from functools import partial
import tqdm

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
		'sentence_boundaries' : tokens.sentences()
    }
    return output


# ------------------------------------------------------------------------------
# Process dataset examples
# ------------------------------------------------------------------------------

def load_data(args, input_file):
	with open(input_file) as f:
		data = json.load(f)['data']
	output = {'qids': [], 'questions': [], 'answers': [],
			  'contexts': [], 'qid2cid': []}
	counter = 0
	for article in data:
		counter += 1
		if args.truncate and counter > 1:
			break
		for paragraph in article['paragraphs']:
			output['contexts'].append(paragraph['context'])
			for qa in paragraph['qas']:
				output['qids'].append(qa['id'])
				output['questions'].append(qa['question'])
				output['qid2cid'].append(len(output['contexts']) - 1)
				if 'answers' in qa:
					output['answers'].append(qa['answers'])
	return output

def main():
	data = load_data(args, args.input_file)

	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

	elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)


	ID_FILE = open(args.output_file + ".json", "w+")
	QUESTION_FILE = open(args.output_file + "_questions.json", "w+")
	PARAGRAPH_FILE = open(args.output_file + "_paragraphs.json", "w+")
	SENTENCES_FILE = open(args.output_file + "_sentences.json", "w+")

	tokenizer_class = tokenizers.get_class(args.tokenizer)
	make_pool = partial(Pool, args.workers, initializer=init)
	workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}, 'classpath' : "/home/bhargavi/robust_nlp/invariance/DrQA/data/corenlp/*"}))
	# workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
	q_tokens = workers.map(tokenize, data['questions'])
	workers.close()
	workers.join()

	workers = make_pool(
		initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'},'classpath' : "/home/bhargavi/robust_nlp/invariance/DrQA/data/corenlp/*"})
		# initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
	)
	c_tokens = workers.map(tokenize, data['contexts'])
	workers.close()
	workers.join()

	job_data = []
	for idx in tqdm(range(len(data['qids']))):
		question = q_tokens[idx]['words']
		question_characters = batch_to_ids([question])
		question_elmo = elmo(question_characters)["elmo_representations"][0]
		# print question id and qidtocid for that question along with embedding information
		dict_ = {"id" : data["qids"][idx], "vectors" : question_elmo.data.numpy().tolist()}
		QUESTION_FILE.write(json.dumps(dict_) + "\n")
	for idx in tqdm(range(len(data["contexts"]))):
		document = c_tokens[idx]['words']
		context_sentence_boundaries = c_tokens[idx]['sentence_boundaries']
		sentences = []
		for s_idx, tup in enumerate(context_sentence_boundaries):
			sentence = document[tup[0]:tup[1]]
			sentences.append(sentence)
		document_characters = batch_to_ids([document])
		sentences_characters = batch_to_ids(sentences)
		document_elmo = elmo(document_characters)["elmo_representations"][0]
		sentences_representations = elmo(sentences_characters)
		sentence_elmo = sentences_representations["elmo_representations"][0]
		sentence_mask = sentences_representations["mask"]
		dict_ = {"id": idx, "vectors": document_elmo.data.numpy().tolist(),
				 "sentence_vectors" : [sentence_elmo.data.numpy().tolist(), sentence_mask.data.numpy().tolist()]}
		QUESTION_FILE.write(json.dumps(dict_) + "\n")
		# write to file with id name, elmo of context and sentences


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--output-file", type=str)
	parser.add_argument("--input-file", type=str)
	parser.add_argument('--workers', type=int, default=None)
	parser.add_argument('--tokenizer', type=str, default='corenlp')
	parser.add_argument("--truncate", action="store_true", default=False)

	args = parser.parse_args()
	main()