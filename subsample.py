import json
import random
import copy

def subsample(data_file, subsample_num):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                all_ids.append(question_answer['id'])

    chosen_ids = set(random.sample(all_ids, k=subsample_num))
    subsampled_data = copy.deepcopy(squad_data)
    # Verify the data
    for article in subsampled_data["data"]:
        for paragraph in article["paragraphs"]:
            new_qas = []
            for qas in paragraph["qas"]:
                if qas['id'] in chosen_ids:
                    new_qas.append(qas)
            paragraph['qas'] = new_qas

    with open(data_file, "w") as output_file:
        json.dump(subsampled_data, output_file)

def count(data_file):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                all_ids.append(question_answer['id'])
    print(len(all_ids))

if __name__ == '__main__':
    dir = 'datasets_subsampled'
    subsample(f'{dir}/indomain_train/nat_questions',1)
    subsample(f'{dir}/indomain_train/newsqa',1)
    subsample(f'{dir}/indomain_train/squad',1)
    subsample(f'{dir}/indomain_val/nat_questions',1)
    subsample(f'{dir}/indomain_val/newsqa',1)
    subsample(f'{dir}/indomain_val/squad',1)