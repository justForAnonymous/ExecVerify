import os

import argparse
import glob
import inspect
import os
import re
import json
import sys
from datetime import datetime
from pydoc import locate
from collections import defaultdict

import numpy as np
import pandas as pd
import pytz
from bullet import Bullet, Input
from tqdm import tqdm

# Import required modules from the project
from inference import Model
from dataset import DREval
from dynamics import Nil, _NilType, FunctionFactory, ClassFactory, Sandbox
from prompt1 import build_direct_prompt, build_cot_prompt
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- Helper Functions ---


def get_time():
    """Get formatted timestamp string in Asia/Shanghai timezone"""
    return datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%y-%m-%d-%H-%M")


def is_builtin_type(cls) -> bool:
    """Check if a class is a built-in type"""
    if cls is None:
        raise ValueError(f"invalid type {cls}")
    assert inspect.isclass(cls), f"Use a class instead of instance: {cls}"
    return cls.__module__ == "builtins"


def penalty_pattern(code: str, _input: str) -> bool:
    """
    Mark failure patterns for Output tasks.
    """
    if (
        "assertTrue(True)" in code
        or "assertFalse(False)" in code
        or "assert True" in code
        or "assert False" in code
        or "assert True == True" in code
        or "assert False == False" in code
    ):
        return True
    given_asserts_num = _input.count("assert")
    if given_asserts_num == 0:
        return False  # If no assert in input, do not apply this penalty
    gen_asserts_num = code.count("assert")
    if gen_asserts_num < given_asserts_num:
        return True
    return False


# --- Core Orchestrator ---


class EvaluationOrchestrator:
    """
    Orchestrator class that coordinates the evaluation process for all tasks.
    1. Prepare prompts for all tasks.
    2. Execute a global batch inference.
    3. Distribute results to each task for evaluation.
    4. Command each task to save results.
    """

    def __init__(self, model_config):
        print("Initializing Evaluation Orchestrator...")
        self.model_config = model_config
        self.model = Model.new(**model_config)
        self.task_data = pd.read_json(f"data/DREval_tasks.jsonl", lines=True).to_dict(
            "records"
        )
        self.tokenizer = None
        model_path = self.model_config.get("model_path")
        if  model_path:
            print(f"Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Add 'mock' attribute to shared model_config for use by Consistency task
        self.model_config["mock"] = True
        # self.tasks = {
        #     "coverage": Coverage(
        #         model=self.model, task_data=self.task_data, **self.model_config
        #     ),
        #     # 'path': Path(model=self.model, task_data=self.task_data, **self.model_config),
        #     # 'state': State(model=self.model, task_data=self.task_data, **self.model_config),
        #     # 'output': Output(model=self.model, task_data=self.task_data, **self.model_config),
        # }
        self.task_names = ["coverage", "path", "state", "output"]
        print("All task objects created.")
        self.system_prompt = """
            A user will ask you to solve a task. You should first draft your thinking process (inner monologue). 
            Then, generate the solution. Your response format must follow the template below:
            <reasoning>
                Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want 
                until you are confident to generate a correct solution.
            </reasoning>
            <answer>
                Final solution presented to the user.
            </answer>
        """
        # self.system_prompt = """You are a programming expert."""

    def gen_batch(self, data: list, batch_size: int) -> list:
        """
        Split a list into fixed-size chunks using list comprehension.

        :param data: Original list
        :param batch_size: Size of each sublist
        :return: A list containing sublists
        """
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    def run_evaluation(self):
        # 1. Preparation phase
        print("\n--- Phase 1: Preparing all prompts from all tasks ---")
        
        
        # self.task_names = ['state']
        for task_name in self.task_names:
            global_prompts = []
            global_contexts = []
            print(f"Preparing prompts for task: {task_name}...")
            task_obj = None
            if task_name == "coverage":
                task_obj = Coverage(
                    model=self.model, task_data=self.task_data, **self.model_config
                )
            if task_name == "path":
                task_obj = Path(
                    model=self.model, task_data=self.task_data, **self.model_config
                )
            if task_name == "state":
                task_obj = State(
                    model=self.model, task_data=self.task_data, **self.model_config
                )
            if task_name == "output":
                task_obj = Output(
                    model=self.model, task_data=self.task_data, **self.model_config
                )

            if task_obj is None:
                print("Not valid task name")
                exit(-1)

            prompts, contexts = task_obj.prepare_all_prompts()
            for context in contexts:
                context["task_name"] = task_name
            global_prompts.extend(prompts)
            global_contexts.extend(contexts)
            print(
                f"Collected {len(prompts)} prompts from {task_name}. Total prompts: {len(global_prompts)}"
            )

            if not global_prompts:
                print("No prompts were generated. Exiting.")
                return

            print(f"\nOriginal number of prompts: {len(global_prompts)}")

            # global_prompts = global_prompts[:128]
            # global_contexts = global_contexts[:128]

            valid_prompts = []
            valid_contexts = []
            for prompt, context in zip(global_prompts, global_contexts):
                if self.tokenizer is None:
                    valid_prompts.append({"role": "user", "content": prompt})
                    valid_contexts.append(context)
                    continue
                # Ensure prompt is a non-empty string
                if prompt and isinstance(prompt, str) and prompt.strip():
                    token_ids = self.tokenizer(prompt).get("input_ids")
                    if token_ids:  # Only valid if tokenization result is non-empty
                        msgs = [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt},
                            # {"role": "assistant", "content": "<reasoning>"},
                        ]
                        valid_prompts.append(
                            self.tokenizer.apply_chat_template(
                                msgs, tokenize=False, add_generation_prompt=True
                            )
                        )
                        valid_contexts.append(context)
                    else:
                        print("here empty prompts")

            print(f"Number of valid (non-empty) prompts: {len(valid_prompts)}")

            if not valid_prompts:
                print("No valid prompts to send to the model. Exiting.")
                return
            # ----------------------------------------------------------------

            # 2. Inference phase
            print("\n--- Phase 2: Running batch inference for all prompts ---")
            print(
                f"Sending {len(valid_prompts)} prompts to the LLM in a single batch..."
            )

            # Use filtered valid_prompts and valid_contexts
            valid_prompts = valid_prompts
            processed_responses = []
            raw_responses = []
            for cur_valid_prompts in self.gen_batch(valid_prompts, 128):
                # print(f'valid_prompts:{valid_prompts[0]}')
                cur_processed_responses, cur_raw_responses = task_obj._prompt_model_batch(cur_valid_prompts)
                processed_responses.extend(cur_processed_responses)
                raw_responses.extend(cur_raw_responses)
            # print(f'raw_responses:{raw_responses[0]}')
            # print(f'processed_responses:{processed_responses[0]}')
            # exit(0)
            # print("Batch inference complete.")

            # processed_responses, raw_responses = self.tasks['path']._prompt_model_batch(valid_prompts)

            # 3. Evaluation and distribution phase
            print("\n--- Phase 3: Distributing results and evaluating ---")
            # Iterate through valid_contexts to ensure one-to-one correspondence
            for p_res, r_res, context, valid_prompt in tqdm(
                zip(processed_responses, raw_responses, valid_contexts, valid_prompts),
                total=len(valid_prompts),
                desc="Evaluating responses",
            ):
                task_name = context["task_name"]
                task_obj.evaluate_and_record_single_response(
                    p_res, r_res, context, valid_prompt
                )
            print("Evaluation and distribution complete.")

            # 4. Save phase
            print(f"\n--- Phase 4: Saving {task_name} results ---")
            task_obj.save_records()
            print(f"Results for task '{task_name}' saved.")


# --- Base Task Class ---


class Task:
    def __init__(
        self, name: str, model: Model, prompt_type: str, task_data: list, **kwargs
    ):
        self.name = name
        self.model = model
        self.prompt_type = prompt_type
        assert prompt_type in ["direct", "cot"], "Use a valid prompt type: direct, cot"
        if not kwargs.get("mock", False):
            assert (
                self.model.prompt_type == prompt_type
            ), "Model prompt type must match task prompt type"
        self.data = pd.read_json(f"data/DREval_data.jsonl", lines=True).to_dict(
            "records"
        )
        self.task_data = task_data
        self.records = []
        self.results_by_task_id = defaultdict(lambda: defaultdict(list))
        self.eval_results = []

    def _get_code(self, idx) -> str:
        return self.data[idx]["code"]

    def _get_entry_point(self, idx) -> str:
        return self.data[idx]["entry_point"]

    def _get_inputs(self, idx) -> str:
        return self.data[idx]["inputs"]

    def _get(self, idx, key) -> str:
        return self.data[idx][key]

    def _build_prompt(self, **kwargs):
        if self.prompt_type == "direct":
            return build_direct_prompt(self.name, **kwargs)
        else:
            return build_cot_prompt(self.name, **kwargs)

    def _prompt_model_batch(self, prompts: list[str]):
        raw_resps = self.model.batch_infer(prompts, self._get_sampling_params())
        processed_resps = [self._postprocess(resp) for resp in raw_resps]
        return processed_resps, raw_resps

    def _extract_xml_answer(self, text: str) -> str:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def _postprocess(self, resp: str):
        raise NotImplementedError()

    @property
    def _metrics(self):
        raise NotImplementedError()

    @property
    def _save_path(self):
        return f"model_generations/{self.name}@{self.model.info}"

    def _get_task_name(self):
        raise NotImplementedError()

    def _get_sampling_params(self):
        raise NotImplementedError()

    def _prepare_prompts_and_contexts_humaneval(
        self, fn_name, code, sub_tasks, sandbox, _input, input_idx
    ):
        raise NotImplementedError()

    def _prepare_prompts_and_contexts_classeval(
        self, test_cls, sub_tasks, _input, input_idx
    ):
        raise NotImplementedError()

    def _evaluate_single_response(self, processed_response, raw_response, context):
        raise NotImplementedError()

    def prepare_all_prompts(self):
        all_prompts = []
        all_contexts = []
        for task in self.task_data:
            idx = task["idx"]
            pairs = task["tasks"]
            if DREval.HUMANEVAL_START <= idx <= DREval.HUMANEVAL_END:
                code = self._get_code(idx)
                fn_name = self._get_entry_point(idx)
                fn = FunctionFactory.create(fn_name, code)
                sandbox = Sandbox(fn)
                inputs = self._get_inputs(idx)
                for pair in pairs:
                    input_idx = pair["input_idx"]
                    _input_val = (
                        inputs[input_idx]
                        if self.__class__.__name__ != "Output"
                        else pair["output_pred"]
                    )
                    prompts_contexts = self._prepare_prompts_and_contexts_humaneval(
                        fn_name, code, pair["task"], sandbox, _input_val, input_idx
                    )
                    for prompt, context in prompts_contexts:
                        all_prompts.append(prompt)
                        all_contexts.append(context)
            elif DREval.CLASSEVAL_START <= idx <= DREval.CLASSEVAL_END:
                cls_code = self._get_code(idx)
                cls_name = self._get_entry_point(idx)
                test_code = self._get(idx, "test")
                ClassFactory.create(cls_name, cls_code)
                test_classes = ClassFactory.create_test_classes(
                    cls_name,
                    cls_code,
                    test_code,
                    DREval.tcls_pattern,
                    DREval.tcls_validation,
                    DREval.tcls_postprocess,
                )
                inputs = self._get_inputs(idx)
                for pair in pairs:
                    input_idx = pair["input_idx"]
                    test_cls = test_classes[input_idx]
                    _input_val = (
                        inputs[input_idx]
                        if self.__class__.__name__ != "Output"
                        else pair["output_pred"]
                    )
                    prompts_contexts = self._prepare_prompts_and_contexts_classeval(
                        test_cls, pair["task"], _input_val, input_idx
                    )
                    for prompt, context in prompts_contexts:
                        all_prompts.append(prompt)
                        all_contexts.append(context)
        return all_prompts, all_contexts

    def evaluate_and_record_single_response(
        self, processed_response, raw_response, context, valid_prompt
    ):
        eval_result = self._evaluate_single_response(
            processed_response, raw_response, context
        )
        eval_result["valid_prompt"] = valid_prompt
        self.eval_results.append(eval_result)
        with open("eval_results.json", "w") as f:
            json.dump(self.eval_results, f)
        # Assume DREval.get_task_id_from_input_idx exists
        task_id_num = context["input_idx"]  # Simplified: assume input_idx is task_id
        task_id = f"DREval/{task_id_num}"
        input_idx = context["input_idx"]
        self.results_by_task_id[task_id][input_idx].append(eval_result)

    def save_records(self):
        for task_id, inputs_data in sorted(self.results_by_task_id.items()):
            generation_records = []
            for input_idx, results in sorted(inputs_data.items()):
                generation_records.append({"input_idx": input_idx, "results": results})
            self.records.append({"task_id": task_id, "generation": generation_records})
        self.records.append(self._metrics)
        print(f"Metrics for {self.name}: {self._metrics}")
        os.makedirs(self._save_path, exist_ok=True)
        with open(f"{self._save_path}/{get_time()}.jsonl", "w+") as f:
            f.writelines([json.dumps(item) + "\n" for item in self.records])
        with open(f"{self._save_path}/{get_time()}_result.jsonl", "w") as f:
            json.dump(self.records[-1], f)

    def run(self):
        print(
            f"Warning: Task.run() is deprecated. Use EvaluationOrchestrator or --task to run evaluations."
        )
        pass


# --- Task Subclasses ---


class Coverage(Task):
    def __init__(self, **kwargs):
        super().__init__("coverage", **kwargs)
        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

    def _acc(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def _prec(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    def _rec(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    def _f1(self):
        prec, rec = self._prec(), self._rec()
        return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    @property
    def _metrics(self):
        return {
            "acc": self._acc(),
            "prec": self._prec(),
            "rec": self._rec(),
            "f1": self._f1(),
        }

    def _postprocess(self, resp: str) -> bool:
        # ans = resp.upper().strip()
        # if self.prompt_type == 'cot' and '[/THOUGHT]' not in ans: return False
        # idx = ans.find('[ANSWER]')
        # if idx != -1: ans = ans[idx+8:].strip()
        # idx = ans.find('[/ANSWER]')
        # if idx != -1: ans = ans[:idx].strip()
        # if ans == '': return False
        # b1, b2 = 'YES' in ans[:3], 'NO' in ans[:3]
        # if b1 == b2: return False
        answer = self._extract_xml_answer(resp)
        answer = answer.upper()
        if "YES" in answer:
            return True
        return False

    def _update_metrics(self, ans, actual):
        if ans and actual:
            self.tp += 1
        elif ans and not actual:
            self.fp += 1
        elif not ans and actual:
            self.fn += 1
        else:
            self.tn += 1

    def _prepare_prompts_and_contexts_humaneval(
        self, fn_name, code, sub_tasks, sandbox, _input, input_idx
    ):
        _, states = sandbox.run(*eval(_input))
        assert (
            sandbox.status == "ok"
        ), f"Error: {sandbox.status} caused by {fn_name}{_input}"
        invocation, codelines = f"{fn_name}{_input[:-2]})", code.split("\n")
        prompts_contexts = []
        for t in sub_tasks:
            line = t["lineno"]
            p = self._build_prompt(
                code=code,
                invocation=invocation,
                invocation_abbr=invocation,
                line=line,
                codeline=codelines[line - 1],
            )
            context = {
                "input_idx": input_idx,
                "expected": states.get_coverage(line - 1),
            }
            prompts_contexts.append((p, context))
        return prompts_contexts

    def _prepare_prompts_and_contexts_classeval(
        self, test_cls, sub_tasks, _input, input_idx
    ):
        obj, setup = test_cls(), ""
        if hasattr(obj, "setUp"):
            obj.setUp()
            setup = "\n# setup code executed previously\n" + "\n# ".join(
                test_cls.__setup__.split("\n")[1:]
            )
            if "Hook method for setting up the test fixture" in setup:
                setup = ""
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == "ok", f"{sandbox.status} caused by code:\n{_input}"
        code, codelines = obj.dreval_test.__doc__, obj.dreval_test.__doc__.split("\n")
        prompts_contexts = []
        for t in sub_tasks:
            line = t["lineno"]
            p = self._build_prompt(
                code=code,
                invocation=setup + "\n" + _input.rstrip(),
                invocation_abbr="the above test code",
                line=line,
                codeline=codelines[line - 1],
            )
            context = {
                "input_idx": input_idx,
                "expected": states.get_coverage(line - 1),
            }
            prompts_contexts.append((p, context))
        return prompts_contexts

    def _evaluate_single_response(self, processed_response, raw_response, context):
        self._update_metrics(processed_response, context["expected"])
        return {
            "generated": raw_response,
            "response": processed_response,
            "expected": context["expected"],
        }

    def _get_task_name(self):
        return "coverage"

    def _get_sampling_params(self):
        return SamplingParams(temperature=0.0, repetition_penalty=1.2, max_tokens=8192,)


class Path(Task):
    def __init__(self, **kwargs):
        super().__init__("path", **kwargs)
        self._correct, self._total = 0, 0

    @property
    def _metrics(self):
        return {"acc": self._correct / self._total if self._total > 0 else 0}

    def _update_metrics(self, ans, actual):
        self._total += 1
        if any(a in actual for a in ans):
            self._correct += 1

    def _postprocess(self, resp: str):
        # if self.prompt_type == 'cot' and '[/THOUGHT]' not in resp: return -2
        # idx = resp.find('[ANSWER]')
        # if idx != -1: resp = resp[idx+8:].strip()
        # idx = resp.find('[/ANSWER]')
        # if idx != -1: resp = resp[:idx].strip()
        # resp = resp.split('\n')[0].strip()
        # if resp == '': return -2
        # elif resp == '-1': return -1
        # else: return resp
        answer = self._extract_xml_answer(resp)
        if answer == "-1":
            return -1
        elif answer == "":
            return -2
        else:
            return answer

    def _prepare_prompts_and_contexts(
        self, code, invocation, invocation_abbr, sub_tasks, states, input_idx
    ):
        codelines = code.split("\n")
        lined_code = "".join([f"{i+1}\t{line}\n" for i, line in enumerate(codelines)])
        prompts_contexts = []
        for t in sub_tasks:
            line = t["lineno"]
            p = self._build_prompt(
                code=lined_code,
                invocation=invocation,
                invocation_abbr=invocation_abbr,
                line=line,
                codeline=codelines[line - 1],
            )
            _actual = states.get_next_line(line - 1)
            actual = [a + 1 if a != -1 else -1 for a in _actual]
            context = {
                "input_idx": input_idx,
                "expected": actual,
                "codelines": codelines,
            }
            prompts_contexts.append((p, context))
        return prompts_contexts

    def _prepare_prompts_and_contexts_humaneval(
        self, fn_name, code, sub_tasks, sandbox, _input, input_idx
    ):
        _, states = sandbox.run(*eval(_input))
        assert (
            sandbox.status == "ok"
        ), f"Error: {sandbox.status} caused by {fn_name}{_input}"
        invocation = f"{fn_name}{_input[:-2]})"
        return self._prepare_prompts_and_contexts(
            code, invocation, invocation, sub_tasks, states, input_idx
        )

    def _prepare_prompts_and_contexts_classeval(
        self, test_cls, sub_tasks, _input, input_idx
    ):
        obj, setup = test_cls(), ""
        if hasattr(obj, "setUp"):
            obj.setUp()
            setup = "\n# setup code executed previously\n# " + "\n# ".join(
                test_cls.__setup__.split("\n")[1:]
            )
            if "Hook method for setting up the test fixture" in setup:
                setup = ""
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == "ok", f"{sandbox.status} caused by code:\n{_input}"
        return self._prepare_prompts_and_contexts(
            obj.dreval_test.__doc__,
            setup + "\n" + _input.rstrip(),
            "the above test code",
            sub_tasks,
            states,
            input_idx,
        )

    def _evaluate_single_response(self, processed_response, raw_response, context):
        ans, actual, codelines = (
            processed_response,
            context["expected"],
            context["codelines"],
        )
        ans_to_lines = []
        if isinstance(ans, int):
            ans_to_lines.append(ans)
        else:
            for i, _line in enumerate(codelines):
                if ans == _line.strip():
                    ans_to_lines.append(i + 1)
            if not ans_to_lines:
                ans_to_lines.append(-2)
        self._update_metrics(ans_to_lines, actual)
        return {"generated": raw_response, "response": ans_to_lines, "expected": actual}

    def _get_task_name(self):
        return "path"

    def _get_sampling_params(self):
        return SamplingParams(temperature=0.0, repetition_penalty=1.2, max_tokens=8192)


class State(Task):
    def __init__(self, **kwargs):
        super().__init__("state", **kwargs)
        self._correct, self._total = 0, 0

    @property
    def _metrics(self):
        return {"acc": self._correct / self._total if self._total > 0 else 0}

    def _update_metrics(self, ans, actual):
        self._total += 1
        if ans == "ERROR":
            return False
        eq = self._eq(ans, actual)
        if eq:
            self._correct += 1
        return eq

    def _eq(self, ans, actual):
        if ans is Nil and actual is Nil:
            return True
        if ans is Nil or actual is Nil:
            return False
        ans_val, ans_type = ans
        if not any(isinstance(a, ans_type) for a in actual if a is not None):
            return False
        if type(ans_val) != ans_type:
            return False
        if ans_type == float:
            return any(abs(ans_val - a) < 1e-6 for a in actual if isinstance(a, float))
        try:
            return ans_val in actual
        except (ValueError, TypeError):
            for a in actual:
                try:
                    if (
                        isinstance(ans_val, np.ndarray)
                        and isinstance(a, np.ndarray)
                        and np.allclose(ans_val, a)
                    ):
                        return True
                    elif ans_val == a:
                        return True
                except Exception:
                    continue
            return False

    def _postprocess(self, resp: str):
        # if self.prompt_type == 'cot' and '[/THOUGHT]' not in resp: return 'ERROR'
        # resp = resp.replace('\u2018', "'").replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"').strip()
        # idx = resp.find('[ANSWER]')
        # if idx != -1: resp = resp[idx+8:].strip()
        # idx = resp.find('[/ANSWER]')
        # if idx != -1: resp = resp[:idx].strip()
        resp = self._extract_xml_answer(resp)
        if resp.capitalize() == "Nil" or resp == "[Nil]":
            return Nil
        semicolon = resp.rfind(";")
        if semicolon == -1:
            return "ERROR"
        val_str, type_str = (
            resp[:semicolon].strip(),
            resp[semicolon + 1 :].strip().lower(),
        )
        if val_str.capitalize() == "Nil":
            return Nil
        if match := re.match(r"<class '(.*)'>", type_str):
            type_str = match.group(1)
        if match := re.match(r"(.*)\[.*\]", type_str):
            type_str = match.group(1)
        type_map = {"string": "str", "integer": "int", "none": "NoneType"}
        type_str = type_map.get(type_str, type_str)
        if "," in type_str or "tuple" in type_str:
            type_str = "tuple"
        try:
            if type_str == "str":
                return eval(val_str), str
            if type_str == "datetime.datetime":
                from dateutil.parser import parse

                return parse(val_str), locate(type_str)
            if type_str in ["numpy.ndarray", "np.ndarray"]:
                return np.array(eval(val_str)), np.ndarray
            if val_str == "None" or type_str == "nonetype":
                return None, type(None)
            _type = locate(type_str)
            if is_builtin_type(_type):
                _val = eval(val_str)
            else:
                _val = _type(eval(val_str))
            return _val, _type
        except Exception:
            return "ERROR"

    def _prepare_prompts_and_contexts(
        self, code, invocation, invocation_abbr, sub_tasks, states, input_idx
    ):
        codelines = code.split("\n")
        prompts_contexts = []
        for t in sub_tasks:
            line, var = t["lineno"], t["var"]
            p = self._build_prompt(
                code=code,
                invocation=invocation,
                invocation_abbr=invocation_abbr,
                line=line,
                codeline=codelines[line - 1],
                var=var,
            )
            context = {
                "input_idx": input_idx,
                "expected": states.interpret_var(line - 1, var),
            }
            prompts_contexts.append((p, context))
        return prompts_contexts

    def _prepare_prompts_and_contexts_humaneval(
        self, fn_name, code, sub_tasks, sandbox, _input, input_idx
    ):
        _, states = sandbox.run(*eval(_input))
        assert (
            sandbox.status == "ok"
        ), f"Error: {sandbox.status} caused by {fn_name}{_input}"
        invocation = f"{fn_name}{_input[:-2]})"
        return self._prepare_prompts_and_contexts(
            code, invocation, invocation, sub_tasks, states, input_idx
        )

    def _prepare_prompts_and_contexts_classeval(
        self, test_cls, sub_tasks, _input, input_idx
    ):
        obj, setup = test_cls(), ""
        if hasattr(obj, "setUp"):
            obj.setUp()
            setup = "\n# setup code executed previously\n# " + "\n# ".join(
                test_cls.__setup__.split("\n")[1:]
            )
            if "Hook method for setting up the test fixture" in setup:
                setup = ""
        sandbox = Sandbox(obj.dreval_test)
        _, states = sandbox.run()
        assert sandbox.status == "ok", f"{sandbox.status} caused by code:\n{_input}"
        return self._prepare_prompts_and_contexts(
            obj.dreval_test.__doc__,
            setup + "\n" + _input.rstrip(),
            "the above test code",
            sub_tasks,
            states,
            input_idx,
        )

    def _evaluate_single_response(self, processed_response, raw_response, context):
        res = self._update_metrics(processed_response, context["expected"])
        return {
            "generated": raw_response,
            "eq": res,
            "expected": str(context["expected"]),
        }

    def _get_task_name(self):
        return "state"

    def _get_sampling_params(self):
        return SamplingParams(temperature=0.0, repetition_penalty=1.2, max_tokens=8192)


class Output(Task):
    def __init__(self, **kwargs):
        super().__init__("output", **kwargs)
        self._total, self._pass = 0, 0

    @property
    def _metrics(self):
        return {"acc": self._pass / self._total if self._total > 0 else 0}

    def _update_metrics(self, status):
        self._total += 1
        if status:
            self._pass += 1

    def _postprocess(self, resp: str):
        return self._extract_xml_answer(resp)

    def _postprocess_phase2(self, resp: str, _input: str):
        if resp == "ERROR":
            return "assert False"
        in_lines, res_lines = _input.strip().split("\n"), resp.strip().split("\n")
        if len(res_lines) >= len(in_lines):
            return resp
        return "\n".join(in_lines[: len(in_lines) - len(res_lines)] + res_lines)

    def _prepare_prompts_and_contexts_humaneval(
        self, fn_name, code, sub_tasks, sandbox, _input, input_idx
    ):
        p = self._build_prompt(code=code, invocation="\n" + _input)
        context = {
            "input_idx": input_idx,
            "type": "humaneval",
            "fn_name": fn_name,
            "code": code,
            "_input": _input,
        }
        return [(p, context)]

    def _prepare_prompts_and_contexts_classeval(
        self, test_cls, sub_tasks, _input, input_idx
    ):
        setup = ""
        if hasattr(test_cls, "setUp"):
            setup = "\n# setup code executed previously\n# " + "\n# ".join(
                test_cls.__setup__.split("\n")[1:]
            )
            if "Hook method for setting up the test fixture" in setup:
                setup = ""
        prelude = "\n# Test code starts here. Only write the completed test code in your answer.\n"
        p = self._build_prompt(
            code=test_cls.__doc__, invocation=setup + prelude + _input
        )
        context = {
            "input_idx": input_idx,
            "type": "classeval",
            "test_cls": test_cls,
            "_input": _input,
        }
        return [(p, context)]

    def _evaluate_single_response(self, processed_response, raw_response, context):
        ans, status = (
            self._postprocess_phase2(processed_response, context["_input"]),
            False,
        )
        if not penalty_pattern(ans, context["_input"]):
            try:
                if context["type"] == "humaneval":
                    locals()[context["fn_name"]] = FunctionFactory.create(
                        context["fn_name"], context["code"]
                    )
                    exec(ans)
                else:  # classeval
                    test_cls = context["test_cls"]
                    fn = FunctionFactory.create_from_answer(ans, test_cls)
                    setattr(test_cls, "dreval_output_pred", fn)
                    obj = test_cls()
                    if hasattr(obj, "setUp"):
                        obj.setUp()
                    obj.dreval_output_pred()
                status = True
            except Exception:
                pass
        self._update_metrics(status)
        return {"generated": raw_response, "pass": status}

    def _get_task_name(self):
        return "output"

    def _get_sampling_params(self):
        return SamplingParams(temperature=0.0, repetition_penalty=1.2, max_tokens=8192)


class Consistency(Task):
    def __init__(self, **kwargs):
        # Consistency doesn't use task_data, so we pop it to avoid super()__init__ error
        # kwargs.pop("task_data", None)
        # super().__init__(name="consistency", **kwargs)
        from types import SimpleNamespace

        # Reconstruct model.info string from config, which is key to finding log files
        model_id = kwargs.get("model_id", "unknown_model")
        prompt_type = kwargs.get("prompt_type", "direct")
        temp = kwargs.get("temp", 0.0)
        
        # Ensure prompt_type is in kwargs to satisfy super().__init__
        if "prompt_type" not in kwargs:
            kwargs["prompt_type"] = prompt_type

        # Simulate the info attribute format from Model class
        mock_model_info_str = f"{model_id}_{prompt_type}_temp{temp}"
        
        # Create a simple "mock model" object with only an info attribute
        mock_model = SimpleNamespace(info=mock_model_info_str)


        super().__init__(name="consistency", model=mock_model, task_data=[], **kwargs)



        self.task_paths = [
            f"model_generations/{task}@{self.model.info}"
            for task in ["coverage", "state", "path", "output"]
        ]
        self.generation_logs = []
        for task_path in self.task_paths:
            try:
                file_path = max(glob.glob(f"{task_path}/*.jsonl"), key=os.path.getctime)
                print(f"Load {file_path}")
                self.generation_logs.append(
                    pd.read_json(file_path, lines=True).to_dict("records")
                )
            except (ValueError, FileNotFoundError):
                print(
                    f"Error: Log file not found in {task_path}. Run main evaluation first.",
                    file=sys.stderr,
                )
                sys.exit(1)

    def _count_statistics(self, task_idx, rule):
        l = []
        for task_log in self.generation_logs[task_idx][
            :-1
        ]:  # Ignore last record (metrics)
            for input_log in task_log.get("generation", []):
                for atomic_log in input_log.get("results", []):
                    # Output task has a nested list structure
                    if self.task_paths[task_idx].startswith("model_generations/output"):
                        # print(f'self.task_paths[task_idx]:{self.task_paths[task_idx]}')
                        # print(f'atomic_log:{atomic_log}')
                        # print(f'atomic_log:{atomic_log.keys()}')
                        atomic_log = atomic_log
                    l.append(rule(atomic_log))
        return l

    def run(self):
        print("Running Consistency evaluation...")
        coverage = self._count_statistics(0, lambda x: x["response"] == x["expected"])
        state = self._count_statistics(1, lambda x: x["eq"])
        path = self._count_statistics(
            2, lambda x: any(y in x["expected"] for y in x.get("response", []))
        )
        output_results = self._count_statistics(3, lambda x: x["pass"])

        # We need to map output results to the other tasks' granularity
        output_expanded = []
        output_idx = 0
        for i, task_log in enumerate(self.generation_logs[0][:-1]):
            for j, input_log in enumerate(task_log["generation"]):
                repeats = len(input_log["results"])
                if output_idx < len(output_results):
                    output_expanded.extend([output_results[output_idx]] * repeats)
                    output_idx += 1

        if not (len(coverage) == len(path) == len(state) == len(output_expanded)):
            print(
                f"Error: Mismatch in log lengths. C:{len(coverage)}, S:{len(state)}, P:{len(path)}, O:{len(output_expanded)}",
                file=sys.stderr,
            )
            return

        total = len(coverage)
        if total == 0:
            print("No data to calculate consistency score.")
            return

        score = sum(
            1
            for i in range(total)
            if coverage[i] and state[i] and path[i] and output_expanded[i]
        )
        score += sum(
            0.5
            for i in range(total)
            if coverage[i] and state[i] and path[i] and not output_expanded[i]
        )
        score += sum(
            0.25
            for i in range(total)
            if coverage[i] and state[i] and not path[i] and not output_expanded[i]
        )
        score += sum(
            0.125
            for i in range(total)
            if coverage[i] and not state[i] and not path[i] and not output_expanded[i]
        )

        print(f"Consistency score: {100 * score/total if total > 0 else 0}")


# --- CLI and Main Execution ---


class Cli:
    def __init__(self):
        self.kwargs = {}
        self._bullet_kwargs = {
            "indent": 0,
            "align": 5,
            "margin": 2,
            "shift": 0,
            "bullet": "\u27f6",
        }

    def get_input(self):
        import readline, glob

        def complete(text, state):
            return (glob.glob(os.path.expanduser(text) + "*") + [None])[state]

        readline.set_completer_delims("\t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(complete)

        # No longer asks for a specific task for the main run
        cli = Bullet(
            prompt="Select prompt type:",
            choices=["direct", "cot"],
            **self._bullet_kwargs,
        )
        self.kwargs["prompt_type"] = cli.launch()
        cli = Bullet(
            prompt="Select model type:",
            choices=["OpenAI", "HuggingFace"],
            **self._bullet_kwargs,
        )
        model_type = cli.launch()
        if model_type == "OpenAI":
            cli = Bullet(
                prompt="Select a model:",
                choices=["gpt-4o-mini", "gpt-4o"],
                **self._bullet_kwargs,
            )
            self.kwargs["model_id"] = cli.launch()
        else:  # HuggingFace
            cli = Bullet(
                prompt="Select deployment type:",
                choices=["Python Instance", "Local API Server"],
                **self._bullet_kwargs,
            )
            if cli.launch() == "Local API Server":
                self.kwargs["port"] = int(
                    Input(
                        prompt="Enter port number: ", default=3000, strip=True
                    ).launch()
                )
            self.kwargs["model_id"] = Input(
                prompt="Enter model name: ", strip=True
            ).launch()
            self.kwargs["model_path"] = input("Enter model path: ")
            self.kwargs["num_gpus"] = int(
                Input(
                    prompt="Enter number of GPUs to use: ", default=1, strip=True
                ).launch()
            )
            default_devices = ",".join(map(str, range(self.kwargs["num_gpus"])))
            ordinals = (
                Input(
                    prompt="Set `CUDA_VISIBLE_DEVICES`: ",
                    default=default_devices,
                    strip=True,
                )
                .launch()
                .split(",")
            )
            self.kwargs["gpu_ordinals"] = [int(_ord) for _ord in ordinals]
        self.kwargs["temp"] = float(
            Input(prompt="Set temperature: ", default=0.0, strip=True).launch()
        )

    @staticmethod
    def config(save_path=".eval_config"):
        cli = Cli()
        cli.get_input()
        with open(save_path, "w+") as f:
            f.write(json.dumps(cli.kwargs, indent=4))
        print(f"Configuration saved to {save_path}")

    def _run(self):
        print(f"The arguments for this run: {self.kwargs}")
        orchestrator = EvaluationOrchestrator(model_config=self.kwargs)
        orchestrator.run_evaluation()

    @staticmethod
    def run_with_config(load_path=".eval_config", task_override=None):
        cli = Cli()
        if not os.path.exists(load_path):
            print(f"Error: {load_path} file not found", file=sys.stderr)
            sys.exit(1)
        with open(load_path, "r") as f:
            cli.kwargs = json.load(f)

        if task_override:
            cli.kwargs["task"] = task_override

        # Route to the correct task runner
        if cli.kwargs.get("task") == "consistency":
            print("Running Consistency task separately...")
            cli.kwargs["mock"] = (
                True  # Ensure consistency doesn't try to init a real model
            )
            task = Consistency(**cli.kwargs)
            task.run()
        else:
            cli._run()  # Run the main orchestrator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation for DREval tasks")
    parser.add_argument(
        "command",
        nargs="?",
        type=str,
        default="run",
        choices=["config", "run"],
        help="Command to run",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=".eval_config",
        help="Specify configuration file to load",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=".eval_config",
        help="Specify configuration file to save",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Specify 'consistency' to run it alone after the main evaluation.",
    )
    args = parser.parse_args()

    if args.command == "config":
        Cli.config(args.output)
    elif args.command == "run":
        Cli.run_with_config(args.input, task_override=args.task)
    else:
        raise RuntimeError("Unreachable")
