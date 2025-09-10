# pip install -U sentence-transformers
import os
import re
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from collections import defaultdict

import torch
import numpy as np
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from retrieval_server import get_retriever, Config as RetrieverConfig
from rerank_server import SentenceTransformerCrossEncoder
from cache import SearchCache
from util import normalize_answer

app = FastAPI()


# ----------- Combined Request Schema -----------
class SearchRequest(BaseModel):
    queries: List[str]
    topk_retrieval: Optional[int] = 10
    topk: Optional[int] = 5
    return_scores: bool = False

# ----------- Reranker Config Schema -----------
@dataclass
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

# ----------- 新增接口 -----------
@app.post("/retrieve/rerank")
def retrieve_with_rerank(request: SearchRequest):
    global retriever, reranker, dataset_name, retriever_topk, cache, use_cache
    normalized_queries = [normalize_answer(query) for query in request.queries]
    # 存储查询结果
    results = []
    # 记录未命中缓存的查询及其索引
    uncached_queries = []
    uncached_indices = []
    
    # 首先检查缓存（可选）
    if use_cache:
        for i, query in enumerate(normalized_queries):
            cached_result = cache.get(query)
            if cached_result is not None:
                # 缓存命中
                results.append(cached_result)
            else:
                # 缓存未命中，添加到待检索列表
                results.append(None)  # 占位
                uncached_queries.append(request.queries[i])
                uncached_indices.append(i)
    else:
        # 不使用缓存时，全部进入检索流程
        results = [None] * len(request.queries)
        uncached_queries = list(request.queries)
        uncached_indices = list(range(len(request.queries)))
    
    # 如果有未缓存的查询，执行检索
    if uncached_queries:
        # Step 1: 检索文档
        retrieved_docs, scores = retriever.batch_search(
            query_list=uncached_queries,
            num=10,
            return_score=True
        )
        
        # Step 2: 一个一个进行重排序
        for i, idx in enumerate(uncached_indices):
            # 对单个查询进行重排序
            single_query = [uncached_queries[i]]
            single_docs = [retrieved_docs[i]]
            reranked = reranker.rerank(single_query, single_docs)
            
            # 处理重排序结果
            doc_scores = reranked.get(0, [])  # 因为只有一个查询，所以用索引0
            doc_scores = doc_scores[:request.topk]
            combined = []
            for doc, score in doc_scores:
                combined.append({"document": doc, "score": score})
            
            # 更新结果
            results[idx] = combined
            
            # 缓存结果
            if use_cache:
                cache.set(normalized_queries[idx], combined)
    
    return {"result": results}

@app.post("/retrieve")
def retrieve_without_rerank(request: SearchRequest):
    global retriever, reranker, dataset_name, retriever_topk, cache, use_cache

    # 存储查询结果
    results = []
    # 记录未命中缓存的查询及其索引
    uncached_queries = []
    uncached_indices = []
    normalized_queries = [normalize_answer(query) for query in request.queries]
    print("request.queries length: ", len(request.queries))
    hit_count = 0
    if use_cache:
        # 首先检查缓存
        for i, query in enumerate(normalized_queries):
            cached_result = cache.get(query)
            if cached_result is not None:
                # 缓存命中
                hit_count += 1
                results.append(cached_result)
            else:
                # 缓存未命中，添加到待检索列表
                results.append(None)  # 占位
                uncached_queries.append(request.queries[i])
                uncached_indices.append(i)
        print("hit_count: ", hit_count)
    else:
        # 不使用缓存时，全部进入检索流程
        results = [None] * len(request.queries)
        uncached_queries = list(request.queries)
        uncached_indices = list(range(len(request.queries)))
    # 如果有未缓存的查询，执行检索
    if uncached_queries:
        # 检索文档并直接返回
        retrieved_docs, scores = retriever.batch_search(
            query_list=uncached_queries,
            num=request.topk,
            return_score=True
        )
        
        # 更新结果并缓存
        for i, idx in enumerate(uncached_indices):
            single_result = retrieved_docs[i]
            score_list = scores[i]
            combined = []
            for doc, score in zip(single_result, scores[i]):
                # 将doc中的'contents'字段改为'content'
                combined.append({"document": doc, "score": score})
            # 更新结果
            results[idx] = combined
            
            # 缓存结果
            if use_cache:
                cache.set(normalized_queries[idx], combined)

    return {"result": results}

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")

    # 基础参数
    parser.add_argument("--index_path", type=str, help="索引文件路径")
    parser.add_argument("--corpus_path", type=str, help="语料库文件路径")
    parser.add_argument("--data_root", type=str, default="/mnt/GeneralModel/zhengxuhui/data/stepsearch", help="数据集根目录")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称，用于缓存和初始化")
    parser.add_argument('--faiss_gpu', action='store_true', help='使用GPU进行计算') 
    parser.add_argument('--port', type=int, default=8000, help='端口地址')
    parser.add_argument('--use_cache', action='store_true', help='是否启用缓存（默认关闭）')

    
    # 检索器配置
    parser.add_argument("--retriever_model", type=str, default="/mnt/GeneralModel/wangziliang1/envs/search_r1/search-model/intfloat/e5-base-v2", help="检索器模型路径")
    parser.add_argument("--topk", type=int, default=10, help="检索的默认topk值")
    parser.add_argument("--retriever_name", type=str, default="e5", help="检索器名称")
    
    # 重排序配置
    parser.add_argument("--reranker_model", type=str, default="/mnt/GeneralModel/wangziliang1/envs/search_r1/search-model/cross-encoder/ms-marco-MiniLM-L12-v2", help="重排序模型路径")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="重排序推理的批处理大小")
    parser.add_argument("--reranking_topk", type=int, default=3, help="每个查询重排序的段落数量")

    args = parser.parse_args()
    
    # 根据数据集名称设置默认值
    if args.dataset_name == "wiki18_e5" or args.dataset_name == "wiki18_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/e5_Flat.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/wiki-18.jsonl"
            
    # elif args.dataset_name == "wiki18_e5_zill" or args.dataset_name == "wiki18_e5_rerank3":
    #     if not args.index_path:
    #         args.index_path = "/mnt/GeneralModel/wangziliang1/data/musi.index"
    #     if not args.corpus_path:
    #         args.corpus_path = "/mnt/GeneralModel/wangziliang1/data/musi.jsonl"
            
    elif args.dataset_name == "musi_e5" or args.dataset_name == "musi_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/musi.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/musi.jsonl"
            
    elif args.dataset_name == "nq_hotpot_e5" or args.dataset_name == "nq_hotpot_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/nq_hotpot.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/nq_hotpot.jsonl"
                        
    elif args.dataset_name == "musi_nq_hotpot_e5" or args.dataset_name == "musi_nq_hotpot_e5_rerank3":
        if not args.index_path:
            args.index_path = f"{args.data_root}/musi_nq_hotpot.index"
        if not args.corpus_path:
            args.corpus_path = f"{args.data_root}/musi_nq_hotpot.jsonl" 

    else:
        print(f"未知的数据集名称: {args.dataset_name}")
        exit(1)
    
    # 1) 初始化检索器配置
    config = RetrieverConfig(
        retrieval_method = args.retriever_name,
        index_path = args.index_path,
        corpus_path = args.corpus_path,
        retrieval_topk = args.topk,
        faiss_gpu = args.faiss_gpu,
        retrieval_model_path = args.retriever_model,
        retrieval_pooling_method = "mean",
        retrieval_query_max_length = 256,
        retrieval_use_fp16 = True,
        retrieval_batch_size = 512,
    )
    
    # 2) 初始化重排序器配置
    reranker_config = RerankerArguments(
        rerank_topk = args.reranking_topk,
        rerank_model_name_or_path = args.reranker_model,
        batch_size = args.reranker_batch_size,
    )
    
    # 3) 实例化全局检索器和重排序器
    retriever = get_retriever(config)
    reranker = get_reranker(reranker_config)
    dataset_name = args.dataset_name
    retriever_topk = args.topk
    use_cache = args.use_cache
    cache = SearchCache(
        dataset_name=dataset_name,
    ) if use_cache else None
    # 4) 启动服务器
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
