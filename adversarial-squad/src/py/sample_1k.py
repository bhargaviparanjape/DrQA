import json
import os
import random
import csv

SEED = 0
random.seed(SEED)

DATA_PATH = "data"
DATA_NAME = "squad"
# FILE_NAME = "dev"
FILE_NAME = "SQuAD-v1.1-addrandom1k-dev"
NUM_DISTINCT_QUESTIONS = 1000


def sample(N):
    file_path = os.path.join(DATA_PATH, DATA_NAME, FILE_NAME + ".json")
    data = json.load(open(file_path, "rb"))

    question_ids = random.sample(list(range(NUM_DISTINCT_QUESTIONS)), N)

    new_data = dict()
    new_data["version"] = data["version"]
    new_data["data"] = list()

    count = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                if int(qas["id"].split("_")[1]) in question_ids:
                    count += 1
                    list_of_titles = [x["title"] for x in new_data["data"]]

                    if article["title"] not in list_of_titles:
                        new_article = dict()
                        new_article["title"] = article["title"]
                        new_data["data"].append(new_article)
                        new_article["paragraphs"] = list()
                        para = dict()
                        para["context"] = paragraph["context"]
                        para["qas"] = list()
                        new_article["paragraphs"].append(para)
                    else:
                        index_in_data = list_of_titles.index(article["title"])
                        para = new_data["data"][index_in_data]["paragraphs"][0]

                    para["qas"].append(qas)

    assert count == N

    with open(os.path.join(DATA_PATH, DATA_NAME, FILE_NAME + "_1k.json"), "w") as f:
        json.dump(new_data, f)


def sample_squad(N):
    file_path = os.path.join(DATA_PATH, DATA_NAME, FILE_NAME + ".json")
    data = json.load(open(file_path, "rb"))

    all_examples = list()

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                example = dict()
                if qas["id"].endswith("high-conf"):
                    example["context"] = paragraph["context"]
                    example["question"] = qas["question"]
                    example["answers"] = qas["answers"]
                    all_examples.append(example)

    samples = random.sample(all_examples, N)

    print(len(all_examples))

    with open(os.path.join(DATA_PATH, DATA_NAME, FILE_NAME + "_50_only_questions.csv"), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, sam in enumerate(samples):
            writer.writerow(["Question-" + str(i), sam["context"], sam["question"]])


if __name__ == "__main__":
    sample_squad(50)
