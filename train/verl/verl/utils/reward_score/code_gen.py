import json
from typing import Any, Dict, Tuple, Union

from verl.utils.reward_score.prime_code import compute_score as prime_code_score


def _load_test_cases(ground_truth: Union[str, Dict[str, Any]]) -> Dict[str, Any]:

    if isinstance(ground_truth, str):
        return json.loads(ground_truth)
    if isinstance(ground_truth, dict):
        return ground_truth
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)}")


def _score_from_pass_rate(pass_rate: float, pass_reward: float, fail_reward: float) -> float:

    pass_rate = max(0.0, min(1.0, float(pass_rate)))
    return pass_rate * pass_reward + (1.0 - pass_rate) * fail_reward

def extract_xml_answer(text: str) -> str:
    if not ("<answer>" in text and "</answer>" in text):
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Dict[str, Any] | None = None,
    *,
    pass_reward: float = 2.0,
    fail_reward: float = 0.0,
) -> Dict[str, Any]:
    format_score = 0.0
    if ("<answer>" in solution_str and "</answer>" in solution_str):
        format_score = format_score + 0.05
    tests = _load_test_cases(ground_truth)
    solution_str = extract_xml_answer(solution_str)
   
    pass_stat, metadata = prime_code_score(
        completion=solution_str,
        test_cases=tests,
        continuous=True,
    )


    if isinstance(pass_stat, bool):
        pass_rate = 1.0 if pass_stat else 0.0
    else:
        pass_rate = float(pass_stat)
        
    if pass_rate!=0.0:
        score = _score_from_pass_rate(pass_rate, pass_reward, fail_reward)
       
    return score+format_score

# verl/utils/reward_score/code_gen.py
def compute_score_prime(data_source, solution_str, ground_truth, extra_info=None):
    return compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        pass_reward=1.0,
        fail_reward=0.0,
    )


def compute_score_raw(
    data_source: str,
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    extra_info: Dict[str, Any] | None = None,
    *,
    pass_reward: float = 2.0,
    fail_reward: float = 0.0,
) -> Dict[str, Any]:
    
    tests = _load_test_cases(ground_truth)
    pass_stat, metadata = prime_code_score(
        completion=solution_str,
        test_cases=tests,
        continuous=True,
    )

    if isinstance(pass_stat, bool):
        pass_rate = 1.0 if pass_stat else 0.0
    else:
        pass_rate = float(pass_stat)
        
    if pass_rate!=0.0:
        score = _score_from_pass_rate(pass_rate, pass_reward, fail_reward)
       
    return score