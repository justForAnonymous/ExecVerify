"""
Extract Trace Dataset - Step 10 (Final) of ExecVerify Data Pipeline

Generates the final multi-task RL dataset by extracting various types of
questions from execution traces (path questions, state questions) and
combining with OI candidates.

Input: io_dataset_for_mutiple_tasks_with_traces.json,
       candidates_oi_for_multi_task_with_cot.json
Output: rl_dataset_multi_task.json
"""

import random
from utils import load_config, setup_logging, load_json, save_json


def append_lineno_str(code_to_run: str) -> str:
    """Add line numbers to code string."""
    code_with_linenos = []
    for i, line_str in enumerate(code_to_run.split("\n")):
        if len(line_str.strip()) > 0:
            code_with_linenos.append(f"{i+1}\t{line_str}")
        else:
            code_with_linenos.append("")
    
    return "\n".join(code_with_linenos)


def get_io_mixed_prompt(func_str: str, func_name: str, func_args: str, questions: list) -> str:
    """
    Build prompt for mixed IO questions (path + state).
    
    Args:
        func_str: Function code
        func_name: Function name
        func_args: Function arguments
        questions: List of questions to ask
        
    Returns:
        Formatted prompt string
    """
    prompt_lines = []
    
    # Intro
    prompt_lines.append(
        "You are a programming expert.\n"
        "Your task is to analyze the Python code and answer the questions. "
        "Mentally simulate the execution; do not actually run any code.\n\n"
        "IMPORTANT — READ THIS:\n"
        'The "Example" section below is for FORMAT REFERENCE ONLY.\n'
        "Do NOT execute the example code and do NOT produce answers for it in your response.\n"
    )
    
    # Example block
    example_block = """
Example (DO NOT EXECUTE OR ANSWER):
Here is the code content:
1    def f(a, b):
2        ans = a + b
3        if ans > 0:
4            ans = int(a)
5        else:
6            ans = b
7        return ans
     
8    assert(f(1.1, 2)) == ????

Here are the questions:
Question1: Fill the assertion statement.
Question2: Is Line 4 (        ans = int(a)) executed when f(1.1, 2) is called? If so, what is the value and type of the variable ans after Line 4 is executed?
Question3: Tracing the call f(1.1, 2), what line is executed immediately after ans = int(a)?

The correct output:
<answer>
assert(f(1.1, 2)) == 1
1; int
return ans
</answer>

— End of Example (DO NOT ANSWER THE EXAMPLE) —
"""
    prompt_lines.append(example_block.strip() + "\n")
    
    # User task
    prompt_lines.append("Your task (this is the ONLY part you should answer):\n")
    
    # Code with assert
    code_to_run = f"""{func_str}

assert {func_name}({func_args}) == ????
"""
    prompt_lines.append("Here is the code content:\n")
    prompt_lines.append(append_lineno_str(code_to_run))
    
    # Questions
    prompt_lines.append(
        "\nHere are the questions (answer in the exact order listed; "
        "if numbers repeat, follow the order of appearance):"
    )
    prompt_lines.append("Question1: Fill the assertion statement.")
    for i, q in enumerate(questions):
        prompt_lines.append(f"Question{i+2}: {q}.")
    
    # Guidelines
    prompt_lines.append("""
Guidelines for "next statement" questions:
- Determine the next line executed after the given statement.
- If the given statement is never reached, answer -1.
- If the program terminates immediately after the given statement, answer -1.
- CRITICAL: Your answer MUST be the exact, verbatim source code of the next line — copied character-for-character, including indentation and punctuation.
- Do NOT include line numbers, quotes, backticks, comments, or any extra words.
- If multiple paths are possible, output any one valid next line.
""".strip())
    
    prompt_lines.append("""
Guidelines for "type & value" questions:
- If the given statement is not executed, answer Nil.
- Otherwise, provide the variable's value and its type.
- If multiple states are possible, output any one valid type and value.
- STRICT FORMAT: value; type
  - Exactly one semicolon and one space.
  - Type must be one of: int, float, bool, complex, str, list, tuple, dict, set, etc.
  - Do NOT add labels, explanations, trailing text, quotes, or backticks.
  - Examples: 1; int    {'a', 'd'}; set    [1, 2]; list
""".strip())
    
    prompt_lines.append("""
ABSOLUTE FORMAT RULES (MUST FOLLOW):
- For assert-statement answers: output the complete assertion statement.
- For "next statement" answers: output ONLY the code statement string. Do not output line numbers!
- For "type & value" answers: output EXACTLY value; type (single semicolon + single space). No extra text.

Output all answers one per line and in the listed order. No bullets, numbering, or additional commentary.
""".strip())
    
    prompt_lines.append("""
Format your response strictly as follows:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
Answer for question1
Answer for question2
...
Answer for questionN.
</answer>
""".strip())
    
    return "\n\n".join(prompt_lines)


def get_oi_prompt(func_str: str, func_name: str, result: str) -> str:
    """Build prompt for OI prediction."""
    return f""""
Fill in the missing assertion. Try to find out the ???? in the following code. 
Here is the provided code: 
```
{func_str}

assert {func_name}(????) == {result}

```

Output the complete test case in the following format:
```python
<test case content, including the function content and the assertion statement>
```

Format your response strictly as follows:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
You answer to the question
</answer>
"""


def get_path_qas(test_case: dict, trace_events: list) -> list:
    """Extract path-related questions from trace events."""
    code_to_run_lines = test_case["code_to_run"].split("\n")
    entry_func_name = test_case["func_name"]
    args = test_case["func_args"]
    
    def get_path_question_str(lineno, exec_time, func_line, func_name, args):
        return (
            f"While executing {func_name}({args}), which line of code is executed "
            f"immediately after line {lineno}: {func_line} has been executed for "
            f"the {exec_time} time?"
        )
    
    qa_map = {}
    time_map = {}
    executed_line_set = set()
    
    for index, trace_event in enumerate(trace_events):
        if (index + 1 == len(trace_events)) or ("line_no" not in trace_event):
            continue
        
        lineno = trace_event["line_no"]
        if lineno not in time_map:
            time_map[lineno] = 1
        else:
            time_map[lineno] = time_map[lineno] + 1
        exec_time = time_map[lineno]
        
        executed_line_set.add(lineno)
        line_str = code_to_run_lines[lineno - 1]
        q = get_path_question_str(lineno, exec_time, line_str, entry_func_name, args)
        
        if line_str.strip().startswith("else") or len(line_str.strip()) <= 5:
            continue
        
        if "line_no" not in trace_events[index + 1]:
            if q not in qa_map:
                qa_map[q] = []
            qa_map[q].append("-1")
            continue
        
        next_lineno = trace_events[index + 1]["line_no"]
        next_line_str = code_to_run_lines[next_lineno - 1]
        
        if q in qa_map and next_line_str not in qa_map[q]:
            qa_map[q].append(next_line_str)
            continue
        
        if (
            ("if" in line_str.strip())
            or ("for" in line_str.strip())
            or ("while" in line_str.strip())
            or ("elif" in line_str.strip())
            or ("if" in next_line_str.strip())
            or ("for" in next_line_str.strip())
            or ("while" in next_line_str.strip())
            or ("elif" in next_line_str.strip())
            or next_lineno != lineno + 1
        ) and (next_lineno != lineno):
            q = get_path_question_str(lineno, exec_time, line_str, entry_func_name, args)
            if q not in qa_map:
                qa_map[q] = []
            if next_line_str not in qa_map[q]:
                qa_map[q].append(next_line_str)
    
    # Add questions for non-executed lines
    max_line = max(list(executed_line_set)) if executed_line_set else 0
    min_line = min(list(executed_line_set)) if executed_line_set else 0
    
    for lineno in range(min_line, len(test_case["func_str"].split("\n")) + 1):
        line_str = code_to_run_lines[lineno - 1]
        if (
            (lineno not in executed_line_set)
            and (len(line_str.strip()) >= 5)
            and (not line_str.strip().startswith("else"))
        ):
            q = get_path_question_str(lineno, 1, line_str, entry_func_name, args)
            qa_map[q] = []
            qa_map[q].append("-1")
    
    return [(q, a) for q, a in qa_map.items()]


def get_state_qas(test_case: dict, trace_events: list) -> list:
    """Extract state-related questions from trace events."""
    code_to_run_lines = test_case["code_to_run"].split("\n")
    entry_func_name = test_case["func_name"]
    args = test_case["func_args"]
    
    def get_state_question_str(lineno, exec_time, line_str, var_name, func_name, args):
        return (
            f"Is line {lineno}: {line_str} executed when {func_name}({args}) is called? "
            f"If so, what are the value and type of the variable '{var_name}' after "
            f"line {lineno} has been executed for the {exec_time} time?"
        )
    
    qa_map = {}
    time_map = {}
    
    for index in range(1, len(trace_events)):
        trace_event = trace_events[index]
        prev_trace_event = trace_events[index - 1]
        
        # Check required keys
        if ("locals" not in trace_event) or ("line_no" not in trace_event) or ("line_no" not in prev_trace_event):
            continue
        
        # Scope isolation: ensure same frame
        curr_frame = trace_event.get("frame_id")
        prev_frame = prev_trace_event.get("frame_id")
        
        if curr_frame is not None and prev_frame is not None and curr_frame != prev_frame:
            continue
        
        lineno = trace_event["line_no"]
        line_str = code_to_run_lines[lineno - 1]
        
        pre_lineno = prev_trace_event["line_no"]
        pre_line_str = code_to_run_lines[pre_lineno - 1]
        
        # Skip loop exit checks
        is_loop_header = "for " in pre_line_str or "while " in pre_line_str
        is_loop_exit = is_loop_header and (lineno != pre_lineno + 1)
        
        if is_loop_exit:
            continue
        
        # Update execution count
        if pre_lineno not in time_map:
            time_map[pre_lineno] = 1
        else:
            time_map[pre_lineno] = time_map[pre_lineno] + 1
        exec_time = time_map[pre_lineno]
        
        # Compare variables
        for var_name, var_val in trace_event["locals"].items():
            flag = False
            if ("locals" not in prev_trace_event) or (var_name not in prev_trace_event["locals"]):
                flag = True
            elif (var_name in prev_trace_event["locals"]) and (prev_trace_event["locals"][var_name] != var_val):
                flag = True
            
            if not flag:
                continue
            
            # Type inference
            try:
                val_obj = eval(var_val, {"__builtins__": {}})
                answer_str = str(val_obj)
                answer_raw_type = str(type(val_obj))
            except:
                answer_str = var_val
                answer_raw_type = "str"
            
            answer_type = ""
            if "int" in answer_raw_type:
                answer_type = "int"
            elif "float" in answer_raw_type:
                answer_type = "float"
            elif "bool" in answer_raw_type:
                answer_type = "bool"
            elif "complex" in answer_raw_type:
                answer_type = "complex"
            elif "str" in answer_raw_type:
                answer_type = "str"
            elif "list" in answer_raw_type:
                answer_type = "list"
            elif "tuple" in answer_raw_type:
                answer_type = "tuple"
            elif "dict" in answer_raw_type:
                answer_type = "dict"
            elif "set" in answer_raw_type:
                answer_type = "set"
            
            if not answer_type:
                answer_type = "object"
            
            q_str = get_state_question_str(
                pre_lineno, exec_time, pre_line_str, var_name, entry_func_name, args
            )
            answer = (answer_type, answer_str)
            
            if q_str not in qa_map:
                qa_map[q_str] = []
            if answer not in qa_map[q_str]:
                qa_map[q_str].append(answer)
    
    return [(q, a) for q, a in qa_map.items()]


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting trace dataset extraction...")
    
    # Load IO dataset with traces
    io_traces_file = config['output_files']['io_traces']
    io_dataset_with_traces = load_json(io_traces_file, logger)
    
    # Extract questions from traces
    logger.info("Extracting questions from traces...")
    rl_dataset = []
    
    for trace_test_case in io_dataset_with_traces:
        try:
            trace_events = trace_test_case["trace_events"]
            test_case = trace_test_case["test_case"]
            
            # Filter by function length or control flow
            if ('for' in test_case["func_str"] or 'while' in test_case["func_str"]) or len(test_case["func_str"].split('\n')) >= 5:
                func_str = test_case['func_str']
                func_name = test_case['func_name']
                func_args = test_case['func_args']
                
                # Skip if function name appears in arguments (recursive cases)
                if func_name in func_args:
                    continue
                
                # Extract path and state questions
                state_qas = get_state_qas(test_case, trace_events)
                random.shuffle(state_qas)
                path_qas = get_path_qas(test_case, trace_events)
                random.shuffle(path_qas)
                
                # Limit questions
                path_qas = path_qas[:5]
                state_qas = state_qas[:5]
                
                # Format questions and answers
                path_qs = []
                path_as = []
                state_qs = []
                state_as = []
                
                for pathq, patha in path_qas:
                    path_qs.append(pathq)
                    path_as.append(patha)
                
                for stateq, statea in state_qas:
                    state_qs.append(stateq)
                    state_as.append(statea)
                
                io_mixed_prompt = get_io_mixed_prompt(
                    func_str, func_name, func_args, path_qs + state_qs
                )
                
                rl_dataset.append({
                    "func_str": test_case["func_str"],
                    "func_name": test_case["func_name"],
                    "func_args": test_case["func_args"],
                    "result": test_case["result"],
                    "prompt_str": io_mixed_prompt,
                    "questions": {"path_questions": path_qs, "state_questions": state_qs},
                    "question_answers": {'path_answers': path_as, 'state_as': state_as},
                    "type": "io_mixed",
                    "trace_events": trace_events
                })
        except Exception:
            pass
    
    logger.info(f"Extracted {len(rl_dataset)} IO mixed samples")
    
    # Limit IO samples
    rl_size = config['dataset_limits']['rl_dataset_size']
    rl_dataset = rl_dataset[:rl_size]
    
    # Load and add OI candidates
    logger.info("Adding OI candidates...")
    oi_candidates_file = config['output_files']['candidates_oi_multi_task']
    oi_candidates = load_json(oi_candidates_file, logger)
    
    oi_size = config['dataset_limits']['oi_rl_dataset_size']
    oi_candidates = oi_candidates[:oi_size]
    
    for sample in oi_candidates:
        rl_dataset.append({
            "func_str": sample["func_str"],
            "func_name": sample["func_name"],
            "func_args": sample["func_args"],
            "result": sample["result"],
            "prompt_str": get_oi_prompt(
                sample["func_str"],
                sample["func_name"],
                sample["result"],
            ),
            "question_answers": {},
            "type": "oi",
        })
    
    logger.info(f"Total dataset size: {len(rl_dataset)}")
    
    # Shuffle dataset
    random.shuffle(rl_dataset)
    
    # Save final dataset
    output_file = config['output_files']['rl_dataset']
    save_json(rl_dataset, output_file, logger)
    
    logger.info("Trace dataset extraction complete!")
    logger.info(f"Final RL dataset: {len(rl_dataset)} samples")


if __name__ == "__main__":
    main()
