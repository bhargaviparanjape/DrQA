import os
import csv
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize

DATA_PATH = "raw_data"
DATA_NAME = "newsqa"
FILE_NAME = "train"
NUM_DISTINCT_STORIES = 11469  # 638 for dev, 637 for test, 11469
NUM_DISTINCT_QUESTIONS = 92549  # 5166 for dev, 5126 for test, 92549 for train

ARTICLE_CACHE = dict()

STORY_ID = 1
STORY_TEXT = 7
OFFSET = 8
QUESTION = 2


def convert_token_to_char_offsets(tokenized_text, token_offsets):
    assert len(token_offsets) == 2
    token_start, token_end = token_offsets

    assert token_start <= len(tokenized_text)
    assert token_end >= token_start

    char_start = accumulate_counts(tokenized_text, start=0, end=token_start)
    char_end = char_start + accumulate_counts(tokenized_text, start=token_start, end=token_end) - 1  # remove last space

    return char_start, char_end


def accumulate_counts(tokenized_text, start, end):
    assert start <= len(tokenized_text)
    assert end >= start
    count = 0
    for i in range(start, end):
        count += len(tokenized_text[i]) + 1
    return count


def main():
    data_path = os.path.join(os.path.abspath("../.."), DATA_PATH, DATA_NAME)
    data_json = dict()
    data_json["version"] = 1
    data_json["data"] = list()

    num_stories = 0
    num_questions = 0

    with open(os.path.join(data_path, FILE_NAME + ".csv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        for i, row in enumerate(reader):
            if i != 0:
                print("Question-", i)
                num_questions += 1
                article_titles = [article["title"] for article in data_json["data"]]

                if row[STORY_ID] not in article_titles:
                    num_stories += 1
                    article = dict()
                    article["title"] = row[STORY_ID]
                    article["paragraphs"] = list()
                    all_qas = dict()
                    all_qas["context"] = row[STORY_TEXT]
                    all_qas["qas"] = list()
                    article["paragraphs"].append(all_qas)
                    data_json["data"].append(article)

                else:
                    index_in_data = article_titles.index(row[STORY_ID])
                    all_qas = data_json["data"][index_in_data]["paragraphs"][0]

                qa = dict()
                qa["question"] = row[QUESTION]
                qa["id"] = "q_" + str(i)

                qa["answers"] = list()
                all_answers = row[8].split(",")
                for answer in all_answers:
                    ans = dict()
                    answer = answer.split(":")
                    start = int(answer[0])
                    end = int(answer[1])
                    ans["answer_start"] = start
                    ans["text"] = row[STORY_TEXT][start:end]
                    qa["answers"].append(ans)

                all_qas["qas"].append(qa)
    
    assert num_questions == NUM_DISTINCT_QUESTIONS
    assert num_stories == NUM_DISTINCT_STORIES

    # print(json.dumps(data_json, indent=2))
    #
    # with open(os.path.join(data_path, FILE_NAME + ".json"), "w", encoding="utf-8") as f:
    #     json.dump(data_json, f)


if __name__ == "__main__":
    main()
