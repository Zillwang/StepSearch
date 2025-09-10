import random
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
import json

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp):
    question = dp['question']

    prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> you are thinking or every time you get new information. \
The question may require you to search for multiple points separately. Please think about which questions or keys that need to be searched and list the questions or keys you plan to search. \
After that, you can search for one question by calling a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want but can only search one question at a time. You might need to rethink about your search questions or the final answer based on search results. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>. The Question you need to answer: {question}\n"""
    prefix = f"""## Background\nYou are a deep AI research assistant. I will give you a single-hop or multi-hop question. \
You don't have to answer the question now, but you should first think about your research plan or what to search for next. \
You can use search to fill in knowledge gaps. \n## Response format: Your output format should be one of the following two formats: \n<think>your thinking process</think>\n\
<answer>your answer after getting enough information</answer>\nor\n<think>your thinking process</think>\nuse <search>search keywords</search> to search for information. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>.\nThe search engine will return the results contained in <information> and </information>. \
\nPlease follow the loop of think, search, information, think, search, information, and answer until the original question is finally solved. \nNote: The retrieval results may not contain the answer or contain noise. \
You need to tell whether there is a golden answer. If not, you need to correct the search query and search again. Question:{question}\n"""
    return prefix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/GeneralModel/zhengxuhui/data/search-r1')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--save_name', type=str, default='hotpot_test')
    parser.add_argument('--answer_name', type=str, default='answer')

    args = parser.parse_args()

    data_source = 'hotpot'

    # Read jsonl file and convert to dataset format
    file_path = f"{args.local_dir}/hotpot_dev_fullwiki_v1.json" 

    # read json file list of dicts
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for i in range(len(data)):
        data[i] = {
            'id': f"hotpot_test_{i}",
            'question': data[i]['question'].strip(),
            'answer': data[i]['answer'],
        }

    
    dataset = {
        'train': datasets.Dataset.from_list(data),
    }

    train_dataset = dataset['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            example['id'] = f"hotpot_test_{idx}"
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['answer'],
                "search_keys": []
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "support_docs": []
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'{args.save_name}.parquet'))
    print(len(train_dataset))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
