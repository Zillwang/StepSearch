import json

data_sets = ['musi', 'hot_dataset', 'hotpot', 'bamboogle']

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def find_question(response):
    if 'The Question you need to answer:' in response:
        question = response.split('The Question you need to answer:')[1].split('\n<|im_end|>\n<|im_start|>')[0].strip().lower()
        return question
    else:
        print(response)
        return None

def prepare_support_docdict(data):
    support_docs = {}
    for item in data:
        sub_support_docs = item['sub_support_docs']
        support_docs[item['id']] = [sub_support_doc['paragraph_text'].lower() for sub_support_doc in sub_support_docs]
    question2id = {item['question'].strip().lower() + ('?' if not item['question'].strip().endswith('?') else ''): item['id'] for item in data}
    return support_docs, question2id

def extract_elements(response):
    elements = {
        'searches': [],
        'informations': [],
        'answer': None
    }
    # response是字符串，匹配其中所有 <search></search>  /<information></information>  /<answer></answer> 标签中的内容
    import re
    searches = re.findall(r'<search>(.*?)</search>', response, re.DOTALL)

    informations = re.findall(r'<information>(.*?)</information>', response, re.DOTALL)
    answer = re.search(r'<answer>(.*?)</answer>', response)
    # informations = [information.split('##>')[1:] for information in informations]
    # 对字符串进行分割，分割标准为 形如 "Doc 1<## Title: XXXX ##>" 的格式(不保留title： XXXX 中 XXXX的内容)
    # informations = [re.split(r'Doc \d+<## Title: .*? ##>', information) for information in informations]
    # # list of list of strings 去除空字符串
    # informations = [[info.strip().lower() for info in infos if info.strip() != ''] for infos in informations]
    # print(informations[0])
    # informations = [information.split('<## Title:')[1:] for information in informations]
    # print(informations[0])
    # informations = [[item.replace('Title:', '').replace('##>', '').strip().lower() for item in information] for information in informations]
    # print(informations[0])
    elements['searches'] = searches
    elements['informations'] =  [item.lower() for item in informations]
    # print(elements['informations'][0])
    # exit()
    elements['answer'] = answer
    return elements

def calculate_metrics(response_data, support_docs, data_set):
    search_count = 0
    hit_count = 0
    usefull_count = 0
    recall_set = set()
    for line in response_data:
        response = line['response']
        elements = extract_elements(response)
        search_count += len(elements['searches'])
        if data_set != 'musi': continue
        information_history = set()
        # print(elements['informations'])
        # exit()
        for information in elements['informations']:
            flag = False
            for info in support_docs[int(line['id'])]:
                if info in information_history:
                    continue
                
                if info in information:
                    hit_count += 1
                    information_history.add(info)
                    recall_set.add(info)
                    flag = True
            if flag:
                usefull_count += 1
        
            
    return search_count, hit_count, usefull_count, recall_set


def main(data_set):
    # 从response_data中挑选 data_source 为 data_set 的response
    partial_response_data = [response for response in response_data if data_set is None or response['data_source'] == data_set]
    search_count, hit_count, usefull_count, recall_set = calculate_metrics(partial_response_data, support_docs, data_set if data_set else 'musi')
    print(f"{data_set} search_count: {search_count}, hit_count: {hit_count}, usefull_count: {usefull_count}, recall_rate: {len(recall_set) / len(support_docs) * 100:.2f}")


if __name__ == '__main__':
    data = read_jsonl('/mnt/GeneralModel/zhengxuhui/data/search-r1/musi_answerable_dev.jsonl')
    support_docs, question2id = prepare_support_docdict(data)
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/steppo_base_3b_test_musique/predictions/20250515_233049.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/steppo_instruct_3b_test_musique/predictions/20250515_233754.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/searchr1_from_huggingface_instruct_vvv10_3b_test_musique/predictions/20250515_225522.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/searchr1_from_huggingface_base_vvv9_3b_test_musique/predictions/20250515_224046.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/zerosearch_base_vvv11_3b_test_musique/predictions/20250515_230228.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/zerosearch_instruct_vvv11_3b_test_musique/predictions/20250515_231322.json')

    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/steppo_base_vvv2_7b_test_musique/predictions/20250515_214252.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/steppo_instruct_7b_test_musique/predictions/20250515_213309.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/research_base_vvv7_7b_test_musique/predictions/20250515_222548.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/research_instruct_vvv8_7b_test_musique/predictions/20250515_223643.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/searchr1_from_huggingface_base_7b_test_all/predictions/20250515_210912.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/searchr1_from_huggingface_instruct_7b_test_all/predictions/20250515_212146.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/zerosearch_base_vvv5_7b_test_musique/predictions/20250515_220713.json')
    # response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/zerosearch_instruct_vvv6_7b_test_musique/predictions/20250515_221719.json')
    response_data = read_json('/mnt/GeneralModel/zhengxuhui/data/search-r1/StepSearch-Base-3B-topk-1/predictions/musi.json')
    response_data = read_json('/mnt/GeneralModel/zhengxuhui/data/search-r1/StepSearch-IT-7B-topk-5/predictions/20250630_214439.json')
    response_data = read_json('/mnt/GeneralModel/ankang/Search-R1/data/no_merged/steppo_base_3b_test_musique/predictions/20250515_233049.json')


    # for data_set in data_sets:
    #     main(data_set)
    main('musi')
