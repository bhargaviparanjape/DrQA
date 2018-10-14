from allennlp.modules.elmo import Elmo, batch_to_ids
import os,sys,argparse,numpy as np,pdb,json
from drqa.reader import utils, config

def main():
	train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
	dev_exs = utils.load_data(args, args.dev_file, skip_no_answer=True)

	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

	elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

	ID_FILE = open(args.output_file + ".txt", "w+")
	QUESTION_FILE = open(args.output_file + "_questions.npy")
	PARAGRAPH_FILE = open(args.output_file + "_paragraphs.npy")
	SENTENCES_FILE = open(args.output_file + "_sentences.npy")
	##  batch this operation
	for exs in train_exs:
		question = exs["question"]
		document = exs["document"]
		id = exs["id"]
		sentences = exs["sentences"]
		question_characters = batch_to_ids(question)
		document_characters = batch_to_ids(document)
		sentences_characters = batch_to_ids(sentences)
		question_elmo = elmo(question_characters)["elmo_representations"]
		document_elmo = elmo(document_characters)["elmo_representations"]
		sentences_representations = elmo(sentences_characters)["elmo_representations"]
		sentence_elmo = sentences_representations["elmo_representations"]
		sentence_mask = sentences_representations["mask"]

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("output-file", type=str)
	parser.add_argument("train-file", type=str)
	parser.add_argument("dev-file", type=str)

	args = parser.parse_args()
	main()