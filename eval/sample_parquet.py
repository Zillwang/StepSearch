import random
import pandas as pd



parquet_path = "/mnt/GeneralModel/zhengxuhui/data/search-r1/ours-test-data-514/merged_data.parquet"
parquet_path = "/mnt/GeneralModel/zhengxuhui/data/search-r1/research-data-512/base-data/merged_data.parquet"
parquet_path = "/mnt/GeneralModel/zhengxuhui/data/search-r1/research-data-512/instruct-data/merged_data.parquet"
parquet_path = "/mnt/GeneralModel/zhengxuhui/data/search-r1/search-r1-baseline-data-511/merged_data.parquet"



set_split = ['musi', 'bamboogle', 'hotpot', 'hot_dataset']
sample_num = 520
seed = 630



# 读取时 不新增空的 index 列
data = pd.read_parquet(parquet_path)
new_data = []

for set_name in set_split:
    set_data = data[data['data_source'] == set_name]
    print(f"{set_name} len: {len(set_data)}")
    # 按照id进行sample
    sample_data = set_data.sample(min(len(set_data), sample_num), random_state=seed)
    new_data.append(sample_data)

new_data = pd.concat(new_data)
print(new_data[new_data['data_source'] == 'bamboogle'].head(10))
out_file = parquet_path.replace('.parquet', f'_sample_{sample_num}.parquet')
new_data.to_parquet(out_file)
