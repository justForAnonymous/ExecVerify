import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

from prompt_exec_verify import crux_input_prompt, crux_output_prompt
from untils import iterating_stops, eval_code, lang_map, input_map


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run CRUXEval-X inference with ExecVerify")
    
    parser.add_argument(
        "--langs",
        type=str,
        default="['java','cpp','cs','go','js','php']",
        help="List of programming languages to evaluate"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directory containing models"
    )
    
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory for CRUXEval-X datasets"
    )
    
    parser.add_argument(
        "--data_input_output",
        type=str,
        required=True,
        help="Directory for preprocessed input/output data"
    )
    
    parser.add_argument(
        "--example_root",
        type=str,
        default=None,
        help="Root directory for examples (if using few-shot)"
    )
    
    parser.add_argument(
        "--example_input_output",
        type=str,
        default=None,
        help="Directory for preprocessed example input/output"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results"
    )
    
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )
    
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel size for vLLM"
    )
    
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=1024,
        help="Maximum number of sequences for vLLM"
    )
    
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=14336,
        help="Maximum model length"
    )
    
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (0-1)"
    )
    
    parser.add_argument(
        "--eval_workers",
        type=int,
        default=32,
        help="Number of worker threads for evaluation"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate"
    )
    
    return parser.parse_args()


def crux_output_prompt_chat(lang, examples, input_text):
    """Chat-style version of crux_output_prompt."""
    problem_str = (
        f"You are given a {lang} code snippet that contains an assert statement with a placeholder `????`.\n"
        "Your goal is to determine what should replace `????`.\n\n"
        "You MUST solve this in two steps:\n"
        "Step 1) Internally derive the COMPLETE assert statement with `????` replaced so that the assert would pass.\n"
        "Step 2) From that completed assert statement, extract ONLY the exact substring/expression that replaces `????`.\n\n"
        "Output requirements:\n"
        "1) In <answer>, output ONLY the extracted replacement for `????` (not the full assert statement).\n"
        "2) Do NOT output any other code (including the assert statement), no code fences, no extra text.\n"
        f"3) The replacement must be valid {lang} syntax for that position and must include any necessary quotes/brackets.\n\n"
        "Response format (MUST match exactly):\n"
        "<reasoning>\n"
        "Briefly show Step 1 and Step 2.\n"
        "</reasoning>\n"
        "<answer>\n"
        "the exact replacement for `????`\n"
        "</answer>"
    )

    message = []
    for example in examples:
        message.append(
            {
                "role": "user",
                "content": f"""{problem_str}

```{lang}
{example["code"].strip()}
```""",
            }
        )
        message.append(
            {
                "role": "assistant",
                "content": f"""<answer>
{example["answer"]}
</answer>""",
            }
        )

    message.append(
        {
            "role": "user",
            "content": f"""{problem_str}

```{lang}
{input_text.strip()}
```""",
        }
    )
    return message


def extract_answer(text: str) -> str:
    """Extract content between <answer>...</answer>; fallback to raw text."""
    text = text.split("</reasoning>")[-1]
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[-1].split("</answer>", 1)[0]
    return text


def build_code_with_mask(lang: str, sample: dict, task_type: str, ds_data_input_output: dict) -> str | None:
    """
    Build code with mask:
    1) Truncate sample["code"] at stop_token
    2) Concatenate preprocessed masked right part
    """
    if "code" not in sample:
        return None

    code = sample["code"]

    # Split using stop token
    flag = False
    for stop_token in iterating_stops[lang]:
        if stop_token in code:
            code = code.split(stop_token)[0]
            flag = True
            break
    if not flag:
        print(f"Warning: No stop token found for sample {sample.get('id')}")
        return None

    # Concatenate masked part
    code_right_part = ds_data_input_output[task_type].get(sample["id"])
    if code_right_part is None:
        print(f"Warning: Missing preprocessed part for sample {sample.get('id')} task={task_type}")
        return None

    return code + code_right_part


def build_prompts(lang: str, ds_data: list, ds_data_input_output: dict, task_type: str, examples: list, tokenizer):
    """
    Returns:
    - cur_prompt: Each element contains {task_id, prompt_messages, code_with_mask}
    - formatted_prompts: String prompts needed by vLLM
    """
    cur_prompt = []
    formatted_prompts = []

    for sample in ds_data:
        code_with_mask = build_code_with_mask(lang, sample, task_type, ds_data_input_output)
        if code_with_mask is None:
            continue

        prompt_messages = crux_output_prompt_chat(lang, examples, code_with_mask)

        # Save code_with_mask to avoid parsing from prompt text later
        cur_prompt.append(
            {
                "task_id": sample["id"],
                "prompt": prompt_messages,
                "code_with_mask": code_with_mask,
            }
        )

        # Properly pass chat messages
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    return cur_prompt, formatted_prompts


def _eval_one(lang: str, code_with_mask: str, gen_text: str) -> bool:
    """
    Single sample evaluation: Replace ???? in code with model output, then call eval_code.
    Returns True if status == OK.
    """
    try:
        answer = extract_answer(gen_text).strip()
        code = code_with_mask.replace("????", answer)
        if "const assert = require('node:assert');" in code:
            code = code.replace("const assert = require('node:assert');", "const assert = require('assert');")
        
        exec_output = eval_code(lang_map[lang], code)
        return exec_output.get("status") == "OK"
    except Exception:
        return False


def eval_batch_multithread(
    lang: str,
    code_with_mask_list: list[str],
    gen_text_list: list[str],
    max_workers: int,
) -> int:
    """
    Multi-threaded batch evaluation, returns number of OK results.
    """
    ok_cnt = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_eval_one, lang, c, t) for c, t in zip(code_with_mask_list, gen_text_list)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"eval[{lang}]"):
            ok_cnt += 1 if fut.result() else 0
    return ok_cnt


def read_data(file_path: str):
    """Read JSON data from file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    
    # Parse languages list
    langs = eval(args.langs) if isinstance(args.langs, str) else args.langs
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct model path
    model_path = os.path.join(args.model_dir, args.model_name) if not os.path.isabs(args.model_name) else args.model_name
    
    print(f"Model Path: {model_path}")
    print(f"Languages: {langs}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Directory: {args.output_dir}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=model_path,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    
    all_log = []
    
    for lang in langs:
        print(f"\n{'='*50}")
        print(f"Processing language: {lang}")
        print('='*50)
        
        log = {"lang": lang}
        
        # Load dataset
        data_file = os.path.join(args.data_root, f"{lang}.json")
        ds_data = read_data(data_file)
        
        # Load preprocessed input_output data
        preprocessed_file = os.path.join(args.data_input_output, f"{lang}.jsonl")
        ds_data_input_output_raw = load_dataset("json", data_files=preprocessed_file)["train"]
        
        ds_data_input_output = {
            "input": {i["id"]: i["input_reasoning"] for i in ds_data_input_output_raw},
            "output": {i["id"]: i["output_reasoning"] for i in ds_data_input_output_raw},
        }
        
        # Load examples (if provided)
        examples = []
        # You can add few-shot examples loading logic here if needed
        
        # Task: output (IO)
        print(f"\nTask: Output (IO)")
        task_type = "output"
        cur_prompt, formatted_prompts = build_prompts(
            lang, ds_data, ds_data_input_output, task_type, examples, tokenizer
        )
        
        print(f"Generating {len(formatted_prompts)} samples...")
        generated_outputs = llm.generate(formatted_prompts, sampling_params)
        
        code_with_mask_list = [p["code_with_mask"] for p in cur_prompt]
        gen_text_list = [g.outputs[0].text for g in generated_outputs]
        
        true_cnt = eval_batch_multithread(lang, code_with_mask_list, gen_text_list, max_workers=args.eval_workers)
        log["io"] = true_cnt / max(1, len(generated_outputs))
        print(f"IO Accuracy: {log['io']:.4f} ({true_cnt}/{len(generated_outputs)})")
        
        # Task: input (OI)
        print(f"\nTask: Input (OI)")
        task_type = "input"
        cur_prompt, formatted_prompts = build_prompts(
            lang, ds_data, ds_data_input_output, task_type, examples, tokenizer
        )
        
        print(f"Generating {len(formatted_prompts)} samples...")
        generated_outputs = llm.generate(formatted_prompts, sampling_params)
        
        code_with_mask_list = [p["code_with_mask"] for p in cur_prompt]
        gen_text_list = [g.outputs[0].text for g in generated_outputs]
        
        true_cnt = eval_batch_multithread(lang, code_with_mask_list, gen_text_list, max_workers=args.eval_workers)
        log["oi"] = true_cnt / max(1, len(generated_outputs))
        print(f"OI Accuracy: {log['oi']:.4f} ({true_cnt}/{len(generated_outputs)})")
        
        all_log.append(log)
        
        # Save results after each language
        output_file = os.path.join(args.output_dir, "results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_log, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("Final Results Summary:")
    print('='*50)
    for log in all_log:
        print(f"{log['lang']}: IO={log['io']:.4f}, OI={log['oi']:.4f}")
    print('='*50)


if __name__ == "__main__":
    main()
