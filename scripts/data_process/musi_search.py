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

    prefix = f"""## Background\nYou are a deep AI research assistant. I will give you a single-hop or multi-hop question. \
You don't have to answer the question now, but you should first think about your research plan or what to search for next. \
You can use search to fill in knowledge gaps. \n## Response format: Your output format should be one of the following two formats: \n<think>your thinking process</think>\n\
<answer>your answer after getting enough information</answer>\nor\n<think>your thinking process</think>\nuse <search>search keywords</search> to search for information. For example, <think> plan to search: (Q1) (Q2) (Q3) ... </think> <search> (Q1) question </search> <think> reasoning ... </think> <answer> Beijing </answer>.\nThe search engine will return the results contained in <information> and </information>. \
\nPlease follow the loop of think, search, information, think, search, information, and answer until the original question is finally solved. \nNote: The retrieval results may not contain the answer or contain noise. \
You need to tell whether there is a golden answer. If not, you need to correct the search query and search again. Question:{question}\n"""
    prefix = f"""## Background\nYou are a deep AI research assistant with search tool\n You should first think about your research plan or what to search for next. \n\n## Response format\n1. You must make search plan inside <plan> and </plan> for in the beginning and after observation.\n2. After plan, if you find you lack some knowledge, you can call a search engine by <search> search keyword </search> and it will return the search results between <information> and </information>.\n3. You must conduct observation inside <observation> and </observation> for EVERY searched document, e.g.<observation>Based on retrieved inforamtion, Doc1 ....; Doc2...; Doc3 ...;</observation>\n4. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. For example, <answer> Beijing </answer>\n\nPlease follow the loop of plan, search, information, observation, plan ... until the you can answer original question.\n Question: {question}\n"""

    return prefix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/mnt/GeneralModel/zhengxuhui/data/stepsearch')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--save_name', type=str, default='musi_stepsearch')
    parser.add_argument('--answer_name', type=str, default='final_answer')
    parser.add_argument('--sample_num', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    data_source = 'musi'

    # Read jsonl file and convert to dataset format
    file_path = f"{args.local_dir}/musi_answerable_{'dev' if args.test else 'train'}.jsonl" 
    
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if args.sample_num is not None:
        random.seed(args.seed)
        data = random.sample(data, args.sample_num)

    
    dataset = {
        'train': datasets.Dataset.from_list(data),
    }

    train_dataset = dataset['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            example['id'] = f"musi_test_{idx}"
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example)
            solution = {
                "target": [example[args.answer_name]] + example['answer_aliases'],
                "search_keys": example['sub_searchs'] if not args.test else []
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
                    "support_docs": example['sub_support_docs'] if not args.test else []
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'{args.save_name}_{"train" if not args.test else "test"}{f"_{args.sample_num}" if args.sample_num is not None else ""}.parquet'))
    print(len(train_dataset))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
