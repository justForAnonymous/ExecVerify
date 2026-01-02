from vllm import LLM, SamplingParams
import json
from transformers import AutoTokenizer
import argparse
from datasets import load_dataset
from prompt_exec_verify import crux_input_prompt, crux_output_prompt
from untils import iterating_stops, eval_code, lang_map, input_map
import json
from tqdm import tqdm
import os


def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_answer(text: str) -> str:
    """Extract content between <answer>...</answer>; fallback to raw text."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[-1].split("</answer>", 1)[0]
    return text


def gen_result(examples, tokenizer, llm, lang, model: str):
    stop = "</answer>"
    if "chat" in model:
        prompts = [
            tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
            for ex in examples
        ]
    else:
        prompts = [ex["prompt"] for ex in examples]

    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8192,
        stop=stop
    )

    print("Sample prompt: {}".format(prompts[0]))
    outputs = llm.generate(prompts, sampling_params)
    for i in range(len(examples)):
        examples[i]["generation"] = outputs[i].outputs[0].text
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", type=str, default="['cpp']")
    parser.add_argument("--tot_data_num", type=int, default=800)
    parser.add_argument("--model_name", type=str, default="WizardCoder-33B-V1.1")
    parser.add_argument("--model_dir", type=str, default=f"./model")
    parser.add_argument("--data_root", type=str, default=f"./datasets/cruxeval-x")
    parser.add_argument("--data_input_output", type=str, default=f"./datasets/cruxeval_preprocessed")
    parser.add_argument("--example_root", type=str, default=f"./datasets/examples")
    parser.add_argument("--example_input_output", type=str, default=f"./datasets/examples_preprocessed")
    parser.add_argument("--output_dir", type=str, default=f"./infer_results")
    parser.add_argument("--codabench", type=bool, default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name_or_path = f"{args.model_dir}/{args.model_name}"
    output_dir = f"{args.output_dir}/{args.model_name}"
    if args.codabench:
        output_dir = f"{args.output_dir}/{args.model_name}_codabench"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Create an LLM.
    llm = LLM(
        model=model_name_or_path,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        max_num_seqs=10,
        max_num_batched_tokens=14336,
        max_model_len=14336,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    langs = eval(args.langs)
    for lang in langs:
        ds_data = read_data(f"{args.data_root}/{lang}.json")
        ds_data = ds_data[:40]
        ds_data_input_output = load_dataset("json", data_files=f"{args.data_input_output}/{lang}.jsonl")["train"]
        example_data = read_data(f"{args.example_root}/{lang}.json")
        example_data_input_output = load_dataset("json", data_files=f"{args.example_input_output}/{lang}.jsonl")["train"]
        ds_data_input_output = {
            "input": {i["id"]: i["input_reasoning"] for i in ds_data_input_output},
            "output": {i["id"]: i["output_reasoning"] for i in ds_data_input_output},
        }
        example_data_input_output = {
            "input": {i["id"]: i["input_reasoning"] for i in example_data_input_output},
            "output": {i["id"]: i["output_reasoning"] for i in example_data_input_output},
        }
        prompt_func = {
            "input": crux_input_prompt,
            "output": crux_output_prompt,
        }
        prompts = {}
        # construct prompt
        for task_type in ["input", "output"]:
            print(task_type)
            examples = []
            for example in example_data:
                code = example["code"]
                flag = False
                for stop_token in iterating_stops[lang]:
                    if stop_token in code:
                        code = code.split(stop_token)[0]
                        flag = True
                        break
                if not flag:
                    assert Exception
                code_right_part = example_data_input_output[task_type][example["id"]]
                examples.append({"code": code + code_right_part, "answer": example[f"{task_type}s"]})

            cur_prompt = []  # {"prompt":prompt,"task_id":task_id}
            for sample in ds_data:
                if "code" not in sample:
                    continue
                code = sample["code"]
                flag = False
                for stop_token in iterating_stops[lang]:
                    if stop_token in code:
                        code = code.split(stop_token)[0]
                        flag = True
                        break
                if not flag:
                    assert Exception
                code_right_part = ds_data_input_output[task_type][sample["id"]]
                code += code_right_part
                cur_prompt.append({"prompt": prompt_func[task_type](lang, examples, code), "task_id": sample["id"]})
            prompts[task_type] = cur_prompt
        # generate answers
        for task_type in ["input", "output"]:
            cur_prompt = prompts[task_type]
            gen_res = gen_result(cur_prompt, tokenizer, llm, lang, "no")
            for cur_res in gen_res:
                cur_res["generation"] = extract_answer(cur_res["generation"])
            # judge the answer
            outputs = [{"id": i, "res": 0} for i in range(args.tot_data_num)]
            for index, cur_res in tqdm(enumerate(gen_res), total=len(gen_res)):
                answer = cur_res["generation"].strip()
                cur_id = cur_res["task_id"]
                # Only keep the last code block content, drop trailing prompt text.
                code_block = cur_prompt[index]["prompt"].split(f"```{lang}")[-1]
                code = code_block.split("```", 1)[0].strip()

                # judge whether calculate the result
                if args.codabench:
                    del outputs[cur_id]["res"]
                    outputs[cur_id]["code"] = code
                    outputs[cur_id]["answer"] = answer
                else:
                    if task_type == "output":
                        code = code.replace("????", answer)
                    else:
                        code = code.replace(input_map[lang], answer)
                    exec_output = eval_code(lang_map[lang], code)
                    if exec_output["status"] != "OK":
                        outputs[cur_id]["res"] = False
                        outputs[cur_id]["error"] = exec_output["status"]
                        outputs[cur_id]["error_message"] = exec_output["stderr"]
                        outputs[cur_id]["code"] = code
                        outputs[cur_id]["answer"] = answer
                    else:
                        outputs[cur_id]["res"] = True
                        outputs[cur_id]["code"] = code
                        outputs[cur_id]["answer"] = answer

            with open(f"{output_dir}/{lang}_{task_type}.json", "w", encoding="utf-8") as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)

