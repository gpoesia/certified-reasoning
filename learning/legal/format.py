import json
import os

TASK = 'diversity_6'
RULE = 'rule.txt'
OPTIONS = 'options.txt'
QUESTION = 'question.txt'

from datasets import load_dataset

train_dataset = load_dataset("nguha/legalbench", TASK, split="train")
test_dataset = load_dataset("nguha/legalbench", TASK, split="test")

def prepare_json(rule, options, question, dataset, file_name):
    json_list = []
    for i in range(len(dataset)):
        data_item = {}
        data_item["rule"] = rule
        data_item["options"] = options
        data_item["scenario"] = dataset[i]["text"] 
        data_item["question"] = question
        if "diversity" in TASK:
            if "yes" in dataset[i]["answer"].lower():
                data_item["answer"] = data_item["options"][0]
            elif "no" in dataset[i]["answer"].lower():
                data_item["answer"] = data_item["options"][1]
            else:
                raise ValueError("Invalid answer")
        else:
            data_item["answer"] = dataset[i]["answer"]        
        json_list.append(data_item)

    # save as jsonl file
    with open(file_name, 'w') as f:
        for item in json_list:
            f.write(json.dumps(item) + "\n")

# read rule
with open(os.path.join('tasks', TASK, RULE), 'r') as f:
    rule = f.read()

# read question
with open(os.path.join('tasks', TASK, QUESTION), 'r') as f:
    question = f.read()

# read options
with open(os.path.join('tasks', TASK, OPTIONS), 'r') as f:
    options = f.read()
    option_list = options.split('\n')

# prepare jsonl file
print(f"Preparing jsonl file...")
print(f"Rule: {rule}")
print(f"Question: {question}")
print(f"Options: {option_list}")
prepare_json(rule, option_list, question, train_dataset, os.path.join('tasks', TASK, 'train.jsonl'))
prepare_json(rule, option_list, question, test_dataset, os.path.join('tasks', TASK, 'test.jsonl'))