from collections import Counter
import json
import string
from typing import List, Dict, Any
import argparse
import re

def extract_answer(text: str) -> str:
    """
    从文本中提取 <answer></answer> 标签对中的内容
    如果不存在正确格式的标签对，返回空字符串
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 标签对中的内容，如果未找到则返回空字符串
    """
    # 查找所有匹配
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    # 如果有匹配，则取最后一个，否则返回空
    if matches:
        matches = matches[-1].strip()
        if "{" in matches:
            matches = matches.split("{")[-1]
            if "}" in matches in matches:
                matches = matches.split("}")[0]
        # print(matches)
        return matches.strip()
    return ""

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def read_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    读取 JSON 文件并返回一个包含所有 JSON 对象的列表
    
    Args:
        file_path (str): JSON 文件的路径
        
    Returns:
        List[Dict[str, Any]]: 包含所有 JSON 对象的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            if not isinstance(data, list):
                data = [data]
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    return data


def exact_match_accuracy(data: List[Dict[str, str]]) -> float:
    """
    计算精确匹配准确率
    """
    if not data:
        return 0.0
    correct = 0
    for item in data:
        response = extract_answer(item["response"])
        ground_truth = item["ground_truth"]
        if type(ground_truth) == str:
            ground_truth = [ground_truth]
        for gt in ground_truth:
            if normalize_answer(response) == normalize_answer(gt):
                correct += 1
                break
        
    return correct / len(data)

def string_f1_score(data: List[Dict[str, str]]):
    def _tokenize(text):
        return text.split()

    def _f1_score(prediction, ground_truth):
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)

        pred_tokens = prediction.split()
        pred_counter = Counter(pred_tokens)
        ans_tokens = ground_truth.split()
        ans_counter = Counter(ans_tokens)

        common = pred_counter & ans_counter
        overlap = sum(common.values())

        if overlap == 0:
            return 0.0

        precision = overlap / len(pred_tokens) if pred_tokens else 0.0
        recall = overlap / len(ans_tokens) if ans_tokens else 0.0

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

    if not data:
        return 0.0
    
    f1_scores = []
    for item in data:
        response = extract_answer(item["response"])
        ground_truth = item["ground_truth"]
        if type(ground_truth) == str:
            ground_truth = [ground_truth]
        f1_scores.append(max(_f1_score(response.lower().strip(), gt.lower().strip()) for gt in ground_truth))

    return sum(f1_scores) / len(f1_scores)


def evaluate_responses(file_path: str, metrics: List[str] = ["exact_match"]) -> Dict[str, float]:
    """
    评估响应质量
    
    Args:
        file_path (str): 数据文件路径
        metrics (List[str]): 要使用的评估指标列表
        
    Returns:
        Dict[str, float]: 各项评估指标的结果
    """
    data = read_json_data(file_path)
    results = {}

    data_sources = set()
    for item in data:
        if 'data_source' in item:
            data_sources.add(item["data_source"])

    
    for data_source in data_sources:
        result = {}
        part_data = [item for item in data if item["data_source"] == data_source]
        if "exact_match" in metrics:
            result["exact_match_accuracy"] = exact_match_accuracy(part_data)
        
        if "f1_score" in metrics:
            result["f1_score"] = string_f1_score(part_data)
        results[data_source] = result

    if len(data_sources) == 0:
        result = {}
        if "exact_match" in metrics:
            result["exact_match_accuracy"] = exact_match_accuracy(data)
        
        if "f1_score" in metrics:
            result["f1_score"] = string_f1_score(data)
        results['all'] = result
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate response quality")   
    parser.add_argument("--file_path", type=str, required=True, help="Path to the evaluation data file")
    parser.add_argument("--metrics", type=str, nargs="+", default=["exact_match", 'f1_score'], help="Metrics to evaluate")
    args = parser.parse_args()

    file_path = args.file_path
    results = evaluate_responses(file_path, args.metrics)
    print("Evaluation Results:")
    for data_source, result in results.items():
        print(f"{data_source}:")
        for metric, score in result.items():
            print(f"{metric}: {score:.4f}")



