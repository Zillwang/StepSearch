import string
import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer

nltk.data.path.append("/mnt/GeneralModel/wangziliang1/envs/search_r1/nltk_data")
for pkg in ["tokenizers/punkt", "tokenizers/punkt_tab",
            "corpora/wordnet", "taggers/averaged_perceptron_tagger",
            "taggers/averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        print(f"❌ {pkg} 缺失")


def normalize_answer(s):
    """规范化文本，去除冠词、标点符号等"""
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


def query_f1_score(prediction, golden_answers):
    """计算查询的F1分数"""
    if prediction is None:
        return 0.0
    prediction = normalize_answer(prediction)
    predictions = [prediction]
    lemmatize_pre = lemmatize_words(prediction)
    if lemmatize_pre != prediction:
        predictions.append(lemmatize_pre)

    if type(golden_answers) == str:
        golden_answers = [golden_answers]
    golden_answers = [normalize_answer(ans) for ans in golden_answers]
    
    extended_answers = golden_answers[:]
    for a in golden_answers:
        lemmatize = lemmatize_words(a)
        if lemmatize != a:
            extended_answers.append(lemmatize)

    max_f1 = 0.0
    for pred in predictions:
        pred_tokens = pred.split()
        pred_counter = Counter(pred_tokens)

        for answer in extended_answers:
            ans_tokens = answer.split()
            ans_counter = Counter(ans_tokens)
            common = pred_counter & ans_counter
            overlap = sum(common.values())
            if overlap == 0:
                continue

            precision = overlap / len(pred_tokens) if pred_tokens else 0.0
            recall = overlap / len(ans_tokens) if ans_tokens else 0.0

            if (precision + recall) == 0:
                current_f1 = 0.0
            else:
                current_f1 = 2 * (precision * recall) / (precision + recall)
            
            max_f1 = max(max_f1, current_f1)
    return round(max_f1, 4)

def lemmatize_words(text):
    """将文本进行词形还原处理"""
    lemmatizer = WordNetLemmatizer()
    try:
        # 分词并标注词性
        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        
        # 将NLTK词性标签转换为WordNet词性标签
        lemmatized_words = []
        for word, tag in tagged_tokens:
            if tag.startswith('J'):
                wordnet_pos = nltk.corpus.wordnet.ADJ
            elif tag.startswith('V'):
                wordnet_pos = nltk.corpus.wordnet.VERB
            elif tag.startswith('R'):
                wordnet_pos = nltk.corpus.wordnet.ADV
            else:  # 默认为名词
                wordnet_pos = nltk.corpus.wordnet.NOUN
                
            lemmatized_words.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
            
        return " ".join(lemmatized_words)
    except Exception as e:
        print(f"词形还原过程出错: {e}")
        return text
