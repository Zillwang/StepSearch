import requests
import json
import tqdm

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

        

def search(queries, topk):
    payload = {
        "queries": queries,
        "topk": topk,
        "return_scores": True
    }
    return requests.post(search_url, json=payload).json()

# search_url = 'http://127.0.0.1:8089/retrieve/rerank'
search_url = 'http://127.0.0.1:8000/retrieve'


train_data = read_jsonl('/mnt/GeneralModel/zhengxuhui/data/search-r1/musi_answerable_train.jsonl')

dev_data = read_jsonl('/mnt/GeneralModel/zhengxuhui/data/search-r1/musi_answerable_dev.jsonl')    

# train_data = read_jsonl('/mnt/GeneralModel/zhengxuhui/data/search-r1/musi_answerable_train_qwen2.5-72b.jsonl')

# dev_data = read_jsonl('/mnt/GeneralModel/zhengxuhui/data/search-r1/musi_answerable_dev_qwen2.5-72b.jsonl')
data = train_data + dev_data
pass_at_k={
    i : 0 for i in range(1, 8)
}
search_count = 0

for line in tqdm.tqdm(data):
    searches = [item for sublist in line['sub_searchs'] for item in sublist]
    search_count += len(searches)
    support_docs = line['sub_support_docs']
    support_docs = [item['paragraph_text'].lower() for item in support_docs]
    for search_key in searches:
        results = search([search_key], 5)['result'][0]
        # 计算pass@1, pass@3, pass@5 results是k个检索结果， support_docs是所有支持文档
        flag_k = {
            i : False for i in range(1, 8)
        }
        for index, result in enumerate(results):
            result = result['document']['content'].lower()
            if result in support_docs:
                flag_k[index+1] = True
        for i in range(1, 8):
            pass_at_k[i] += flag_k[i]
            

print(pass_at_k)
for i in range(1, 8):
    if i>1:
        pass_at_k[i] += pass_at_k[i-1]
    print(pass_at_k[i]/search_count)
