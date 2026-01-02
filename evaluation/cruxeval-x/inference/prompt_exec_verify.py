def _fmt_examples(lang, examples):
    """Format few-shot examples with reasoning/answer tags."""
    example_str = ""
    for example in examples:
        example_str += f"""```{lang}
{example["code"].strip()}
```
<answer>
{example["answer"]}
</answer>
"""
    return example_str.rstrip()


def crux_input_prompt(lang, examples, input_text):
    """
    Given a function and its expected output, find an input (argument list or
    function call) so that the assertion would pass. Model should ONLY output
    the missing input expression (the replacement for "????"), not the assert.
    """
    example_str = _fmt_examples(lang, examples)
    return f"""Fill in the missing input so that the check would pass. You are given a {lang} function and the expected output. Find an input (as a function call or arguments) such that executing the function produces that output. Do NOT output the assert statement or any code fences. Output ONLY the missing expression that replaces "????".


```{lang}
{input_text}
```

Format your response strictly as:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the expression that replaces ????
</answer>
"""


def crux_output_prompt(lang, examples, input_text):
    """
    Given a function and a test case with an incomplete assert, predict the
    correct output value. Model should ONLY output the value that replaces
    "????", not the full assert.
    """
    example_str = _fmt_examples(lang, examples)
    return f"""Fill in the missing output for the assert. You are given {lang} code with an assert containing "????". Find the value that should replace "????" so the assert passes. Do NOT output the assert statement or any code fences. Output ONLY the missing value.


```{lang}
{input_text}
```

Format your response strictly as:
<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the value that replaces ????
</answer>
"""


def crux_input_prompt_chat(lang, examples, input_text, model: str):
    """Chat-style version of crux_input_prompt."""
    problem_str = (
        f"Fill in the missing input so that the check would pass. You are given a "
        f"{lang} function and the expected output. Find an input (as a function "
        f"call or arguments) such that executing the function produces that "
        "output. Do NOT output the assert statement or any code fences. Output "
        'ONLY the missing expression that replaces "????".\n\n'
        "Format your response strictly as:\n"
        "<reasoning>\n"
        "your step-by-step reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "the expression that replaces ????\n"
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
        message.append({"role": "assistant", "content": f"""<answer>
{example["answer"]}
</answer>"""})

    message.append(
        {
            "role": "user",
            "content": f"""{problem_str}

```{lang}
{input_text}
```""",
        }
    )
    return message


def crux_output_prompt_chat(lang, examples, input_text, model: str):
    """Chat-style version of crux_output_prompt."""
    problem_str = (
        f"Fill in the missing output for the assert. You are given {lang} code "
        'with an assert containing "????". Find the value that should replace '
        '"????" so the assert passes. Do NOT output the assert statement or any '
        "code fences. Output ONLY the missing value.\n\n"
        "Format your response strictly as:\n"
        "<reasoning>\n"
        "your step-by-step reasoning here\n"
        "</reasoning>\n"
        "<answer>\n"
        "the value that replaces ????\n"
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
        message.append({"role": "assistant", "content": f"""<answer>
{example["answer"]}
</answer>"""})

    message.append(
        {
            "role": "user",
            "content": f"""{problem_str}

```{lang}
{input_text}
```""",
        }
    )
    return message

