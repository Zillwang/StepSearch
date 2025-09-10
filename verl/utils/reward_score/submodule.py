from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def subm_tfidf_cosine(input_str: str, concept_units: list[str]) -> float:
    """
    基于TF-IDF余弦相似度的最大覆盖子模函数。
    
    Args:
        input_str (str): 当前的输入字符串。
        concept_units (List[str]): 概念单元列表。
        
    Returns:
        float: 总得分。
    """
    if input_str == '':
        return [0.0] * len(concept_units)
    
    # 建立文档集：input_str + concept_units
    documents = [input_str] + concept_units
    
    # TF-IDF编码
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 计算input与每个concept_unit的相似度
    input_vec = tfidf_matrix[0]  # 第一行是input
    concept_vecs = tfidf_matrix[1:]  # 后面是concepts
    
    # 两两计算余弦相似度
    similarities = cosine_similarity(input_vec, concept_vecs).flatten()
    
    # 只保留非负值，累加
    total_score = np.maximum(similarities, 0)
    
    return total_score