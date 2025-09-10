# console 传入参数 
file_path=${1:-"/mnt/GeneralModel/zhengxuhui/data/search-r1/Search-R1-Base-7B-Web/predictions/20250702_060520.json"}
metrics=${2:-"exact_match f1_score"}

python scripts/eval.py --file_path $file_path --metrics $metrics