import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
import re

def extract_solution(solution_str):
    def extract_xml_answer(text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_python_code(text:str) -> str:
        pattern = r"```python\n(.*?)\n```"
        matches = re.findall(pattern, text, re.DOTALL)
        if len(matches) <=0:
            return "assert False"
        return matches[0]
    
    return extract_xml_answer(extract_python_code(solution_str))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/code_exec")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    data_source = '' # set the data source path
    dataset = datasets.load_dataset("json",data_files=data_source)

    train_dataset = dataset["train"]
    test_dataset = train_dataset.select([0])

    

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example['prompt_str']
            data = {
                "data_source": data_source,
                "prompt": [
                    {   "role":"system",
                        "content": "You are a programming expert.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": "",
                    "question": question,
                    "ground_truth": "",
                    "func_str": example["func_str"],
                    "func_name": example["func_name"],
                    "func_args": example["func_args"],
                    "result": example["result"],
                    "prompt_str": example['prompt_str'],
                    "question_answers":example['question_answers'],
                    "type": example['type'],
                },
            }
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "rl_dataset_train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "rl_dataset_test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
