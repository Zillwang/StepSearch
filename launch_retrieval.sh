data_root=/mnt/GeneralModel/zhengxuhui/data/stepsearch
dataset_name=musi_e5

python search_r1/search/retrieval_rerank_server.py \
    --data_root $data_root \
    --faiss_gpu \
    --port 8000 \
    --dataset_name $dataset_name