import json
import random
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time
import argparse
# 新搜索服务所使用的配置

API_KEY = 'your api key'
args = argparse.ArgumentParser()
args.add_argument("--api_key", type=str, default=API_KEY)
args.add_argument("--port", type=int, default=8000)
args.add_argument("--host", type=str, default="0.0.0.0")
args = args.parse_args()

SEARCH_URL = "https://google.serper.dev/search"
HEADERS = {
    'X-API-KEY': args.api_key,
    'Content-Type': 'application/json'
}

app = FastAPI()

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

def new_search(query: str, topk: int) -> List[dict]:
    """
    对单个查询调用新搜索接口，并返回转换后的搜索结果列表
    每个结果包含随机生成的 id、由 title 和 snippet 拼接后的内容，以及随机生成的 score。
    当发生网络错误或解析错误时，返回一个带有错误提示的结果列表。
    """
    try:
        payload = json.dumps({"q": query})
        response = requests.post(SEARCH_URL, headers=HEADERS, data=payload)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Error during search for query '{query}': {e}")
        document = {
            "id": str(random.randint(1, 6553555)),
            "contents": "Search failed, please search again\nSearch failed, please search again"
        }
        score = round(random.uniform(0, 1), 10)
        return [{
            "document": document,
            "score": score
        }]

    results = []
    count = 0
    for item in data.get("organic", []):
        if count >= topk:
            break
        document = {
            "id": str(random.randint(1, 6553555)),
            "title": item.get("title", ""),
            "content": item.get("snippet", ""),
        }
        score = round(random.uniform(0, 1), 10)
        results.append({
            "document": document,
            "score": score
        })
        count += 1
    return results

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    请求示例：
    {
      "queries": ["boy", "girl"],
      "topk": 2,
      "return_scores": true
    }
    对每个查询调用 new_search。若 return_scores 为 True，则返回 document 和 score，
    否则只返回 document。
    每个查询独立执行，即使某个查询出现错误也不影响其它查询。
    """
    # 默认 topk 值，如果请求中没有提供，则默认为 10
    topk = request.topk if request.topk is not None else 10
    resp = []
    for query in request.queries:
        time.sleep(0.2)
        try:
            search_results = new_search(query, topk)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            # 构造一个错误提示结果
            document = {
                "id": str(random.randint(1, 6553555)),
                "contents": "Search failed, please search again"
            }
            score = round(random.uniform(0, 1), 10)
            search_results = [{
                "document": document,
                "score": score
            }]
        if request.return_scores:
            resp.append(search_results)
        else:
            resp.append([item["document"] for item in search_results])
    return {"result": resp}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
