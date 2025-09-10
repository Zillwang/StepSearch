from collections import Counter
import re
import string
import random

from verl.utils.reward_score.extract_tags import extract_prompt_tags
from verl.utils.reward_score.submodule import subm_tfidf_cosine


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

def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()
    
def extract_titles_from_information(information):
    """
    Extract all text between <## and ##> tags from the information string.
    
    Args:
        information: String containing text with <## text ##> tags
        
    Returns:
        List of extracted texts between tags
    """
    pattern = r'<##(.*?)##>'
    matches = re.findall(pattern, information)
    return [match.replace('Title:', '').strip() for match in matches]

def extract_content_from_information(information):
    """
    Extract the content after each <## text ##> tag from the information string.
    
    Args:
        information: String containing text with <## text ##> tags
        
    Returns:
        List of String after each <## text ##> tag
    """
    information = information.split('##>')[1:]
    information = [item.split('<##')[0].strip() for item in information]
    return information


def query_f1_score(prediction, golden_answers):
    def _tokenize(text):
        return text.split()

    if prediction is None:
        return 0.0
    prediction = normalize_answer(prediction)
    pred_tokens = prediction.split()
    pred_counter = Counter(pred_tokens)

    if type(golden_answers) == str:
        golden_answers = [golden_answers]
    golden_answers = [normalize_answer(answer) for answer in golden_answers]

    max_f1 = 0.0
    for answer in golden_answers:
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

def step_information_gain(informations:list[list[str]], golden_info:list[dict])->list[float]:
    """
    Calculate the information gain for each information in the list.
    
    Args:
        informations: List of list of String
        golden_info: List of dicts with keys 'title' and 'paragraph_text'

    Returns:
        List of information gain scores
    """

    if len(golden_info) == 0:
        return [], []
    # Normalize golden info
    golden_infos = [normalize_answer(info['paragraph_text']) for info in golden_info]
    previous_match_degree = [0.0 for _ in golden_infos]

    # Calculate information gain for each information
    information_gains = []
    for information in informations:
        current_match_degree = [0.0 for _ in golden_infos]
        # Normalize information
        for info in information:
            info = normalize_answer(info)

            # Calculate information gain
            for i, score in enumerate(subm_tfidf_cosine(input_str=info, concept_units=golden_infos)):
                current_match_degree[i] = max(current_match_degree[i], score)

        information_gains.append(
            sum(max(current_match_degree[i] - previous_match_degree[i], 0) for i in range(len(current_match_degree)))
            / len(current_match_degree)
            )

        previous_match_degree = [max(current_match_degree[i], previous_match_degree[i]) for i in range(len(current_match_degree))]
    
    redundancy_penalty = [0.0 for _ in information_gains]
    info_gotten = set()
    for i, information in enumerate(informations):
        for info in information:
            if info in info_gotten:
                redundancy_penalty[i] += 1/len(information)
            else:
                info_gotten.add(info)
    
    
    return information_gains, redundancy_penalty

def answer_last_check(text: str):
    # 匹配 answer 区块，非贪婪模式，支持跨行
    pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)
    matches = list(pattern.finditer(text))
    
    if not matches:
        # 未找到任何 <answer>...</answer>
        return False
    
    # 取最后一个 match
    last_match = matches[-1]
    start, end = last_match.span()
    
    # 区块之后的文本
    trailing_text = text[end:]
    # 判断去除所有空白字符后是否还有内容
    if len(trailing_text.strip()) > 0: return False
    
    return True


def step_search_keys_match(searches, keys):
    """Check if the search keys match the golden keys."""
    """
    Args:
        searches: List of search queries
        keys: List of lists where each sublist contains different expressions for a search key
        
    Returns:
        float: Score between 0 and 1 indicating match quality
    """
    
    keys = [list(arr) for arr in keys]
    if not searches or not keys:
        return 0.0
        
    # Track which key groups have been matched
    matched_key_groups = {i:0 for i in range(len(keys))}
    total_matches = 0
    
    # Process each search query
    for search in searches:
        best_match_score = 0
        best_match_group = -1
        
        # Compare against each key group
        for group_idx, key_group in enumerate(keys):
            match_score = query_f1_score(prediction=search, golden_answers=key_group)

            # print(f"search: {search},  match_score: {match_score}")

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_group = group_idx
        
        # Update the matched key groups
        matched_key_groups[best_match_group] = best_match_score

    total_matches = sum(matched_key_groups.values())
                
    # Return proportion of key groups that were matched
    return total_matches / len(keys)

def compute_score_f1_steps_plan_with_support_docs(config, solution_str, ground_truth, method='strict', support_docs=None, format_score=0.0, search_keys_score=0.618):
    # Find <information> content that follows each <search> tag

    final_score = 0.0
    answer = extract_solution(solution_str=solution_str)

    answer_correct = query_f1_score(prediction=answer, golden_answers=ground_truth['target'])

    if answer_last_check(solution_str):
        final_score += answer_correct
    else:
        return {
            'score': -1,
            'answer_correct': answer_correct,
            'step_scores': [],
            'search_key_score': 0.0,
        }
                

    information_matches = re.finditer(r'<search>.*?</search>\s*<information>(.*?)</information>', solution_str, re.DOTALL)
    information_matches = [
        # extract_titles_from_information(match.group(1).strip())
        extract_content_from_information(match.group(1).strip())
        for match in information_matches
    ]

    information_gains, redundancy_penalty = step_information_gain(informations=information_matches, golden_info=support_docs)
    step_scores = [(gain if config.trainer.information_gain and config.trainer.search_steps_reward else 0.0) - (penalty if config.trainer.redundancy_penalty and config.trainer.search_steps_reward else 0.0) for gain, penalty in zip(information_gains, redundancy_penalty)]


    searches = [re.sub(r'</?search>', '', match).strip() for match in re.findall(r'<search>.*?</search>', solution_str, re.DOTALL)]

    search_key_score = 0.0
    if 'search_keys' in ground_truth:
        search_key_score = step_search_keys_match(searches=searches, keys=ground_truth['search_keys'])

    do_print = random.randint(1, 64) == 1

    if config.trainer.search_key_reward:
        final_score += search_key_score * search_keys_score

    if do_print:
        print(f"----------------rm_f1_steps_plan_with_support_docs----------------")
        print(f"Solution string: {solution_str}")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Answer correct(f1): {answer_correct}")
        print(f"Information gains: {information_gains}")
        print(f"Redundancy penalty: {redundancy_penalty}")
        print(f"Step scores: {step_scores}")
        print(f"Search key score: {search_key_score}")

    return {
        'score': final_score,
        'answer_correct': answer_correct,
        'step_scores': step_scores,
        'search_key_score': search_key_score,
    }