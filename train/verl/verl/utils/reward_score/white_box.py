import subprocess
import sys
import re
from transformers import AutoTokenizer
import ast


def run_python_code_reward(code_string):
    try:
        process = subprocess.run(
            [sys.executable, '-c', code_string],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False 
        )

        if process.returncode == 0:
            return True

        if 'AssertionError' in process.stderr:
            return False
    except Exception as e:
        return False

    return False

def extract_xml_answer(text: str) -> str:
        if not ("<answer>" in text and "</answer>" in text):
            return ""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

def extract_solution(solution_str):
    
    def extract_python_code(text:str) -> str:
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) <=0:
            return "assert False"
        return matches[0]
    
    return extract_python_code(extract_xml_answer(solution_str))


def safe_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        # If it's not a literal (e.g., a simple string that's not quoted), return as is
        return value

def cal_path_true_cnt(path_answers,path_gts):
    true_cnt = 0
    for path_answer,path_gt in zip(path_answers,path_gts):
        for cur_gt in path_gt:
            if path_answer.strip() == cur_gt.strip():
                true_cnt = true_cnt + 1
                break

    return true_cnt

def cal_state_true_cnt(state_answers,state_gts):
    true_cnt = 0
    for state_answer,state_gt in zip(state_answers,state_gts):
        try:
            type_str = state_answer.split(';')[1].strip()
            value_str = state_answer.split(';')[0].strip()

            value = safe_eval(value_str)
            for cur_gt in state_gt:
                cur_gt_type = cur_gt[0].strip()
                if cur_gt_type == 'str':
                    cur_gt_value = eval(repr(cur_gt[1]))
                else:
                    cur_gt_value = eval(cur_gt[1])

                if type_str == cur_gt_type.strip() and value == cur_gt_value:
                    true_cnt = true_cnt +1
                    break
            
        except Exception as e:
            continue
    return true_cnt

def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    if extra_info['type'] == 'oi':
        python_code = extract_solution(solution_str)
        if run_python_code_reward(python_code):
            return 2.0
        else:
            return 0.0
    elif extra_info['type'] == 'io_mixed':
        func_str = extra_info["func_str"]
        func_name = extra_info["func_name"]
        func_args = extra_info["func_args"]
        result = extra_info["result"]

        answer = extract_xml_answer(solution_str)
        if len(answer)==0:
            return 0.0
        
        assert_stmt = answer.split('\n')[0].strip()
        other_answers = answer.split('\n')[1:]

        if (func_args not in assert_stmt) or not (assert_stmt.startswith('assert')):
            return 0.0

        code_to_run = f"""
{func_str}

{assert_stmt.strip()}
"""    
        if not run_python_code_reward(code_to_run):
            return 0.0

        cur_reward = 1.0

        question_answers = extra_info["question_answers"] # :{'path_answers':path_as,'state_as':state_as},

        if len(question_answers) == 0:
            return cur_reward + 1.0

        path_gts = question_answers['path_answers']
        state_gts = question_answers['state_as']

        if len(other_answers)!=len(path_gts)+len(state_gts) or len(path_gts)+len(state_gts)==0:
            return cur_reward
        
        path_answers = other_answers[:len(path_gts)]
        state_answers = other_answers[len(path_gts):]


        answer_true_cnt = cal_path_true_cnt(path_answers,path_gts) + cal_state_true_cnt(state_answers, state_gts)
        den = len(path_gts) + len(state_gts)
        wb_acc = (answer_true_cnt / den) if den > 0 else 0.0  # in [0,1]

        io = float(cur_reward)  
        alpha = 0.50    

        return 2.0 * ((1.0 - alpha) * io + alpha * wb_acc)  # in [0,2]
        
    return 0.0

