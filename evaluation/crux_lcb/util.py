"""
Utility functions for code execution evaluation.

This module provides functions for:
- Parsing Python code using tree-sitter
- Extracting function names, arguments, and results
- Generating evaluation prompts
- Testing code execution
"""

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import sys
import subprocess


def extract_xml_answer(text: str) -> str:
    """
    Extract answer content from XML-like tags.
    
    Args:
        text: Text containing <answer>...</answer> tags
        
    Returns:
        str: Extracted answer content
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_args(test_str):
    """
    Extract function arguments from assertion statement.
    
    Args:
        test_str: Python code string containing assertion
        
    Returns:
        str: Extracted argument list as string
    """
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(bytes(test_str, "utf8"))
    root_node = tree.root_node

    args_query_string = """
    (
        assert_statement
        (
            comparison_operator(
                call
                    function:(identifier) @func_name
                    arguments: (argument_list) @args
                )
        )
    )
    """
    args_query = language.query(args_query_string)
    args_captures = args_query.captures(root_node)

    args = []
    if "args" in args_captures:
        args = args_captures["args"]
    if len(args) != 1:
        return ""

    return args[0].text.decode("utf8")


def extract_func_name(func_str):
    """
    Extract function name from function definition.
    
    Args:
        func_str: Python function definition string
        
    Returns:
        str: Function name
    """
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(bytes(func_str, "utf8"))
    root_node = tree.root_node

    def_query_string = """
    (   
        function_definition
            name:(identifier)@func_name 
    ) @func_def
    """
    def_query = language.query(def_query_string)
    def_captures = def_query.captures(root_node)
    func_names = []
    if "func_name" in def_captures:
        func_names = def_captures["func_name"]
    return func_names[0].text.decode("utf8")


def extract_result(test_str):
    """
    Extract expected result from assertion statement.
    
    Args:
        test_str: Python code string containing assertion
        
    Returns:
        str or None: Extracted result value
    """
    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(bytes(test_str, "utf8"))
    root_node = tree.root_node
    
    assert_query_string = """
    (
        assert_statement
        (
            comparison_operator 
        )@assert_st
    )
    """
    assert_query = language.query(assert_query_string)
    assert_captures = assert_query.captures(root_node)
    
    try:
        result_str = assert_captures['assert_st'][0].children[2].text.decode('utf8')
        return result_str
    except Exception:
        return None


def get_prompt_str(func_str, func_name, exec_result, args="????"):
    """
    Generate prompt for assertion completion task.
    
    Args:
        func_str: Function code string
        func_name: Function name
        exec_result: Expected execution result
        args: Function arguments (use "????" for unknown)
        
    Returns:
        str: Formatted prompt string
    """
    return f"""Fill in the missing assertion. Try to find out the ???? in the following code. 
Here is the provided code: 
```
{func_str}

assert {func_name}({args}) == {exec_result}

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


def generate_r1_prompt(tokenizer, func_str, exec_result, args_str="????"):
    """
    Generate formatted prompt using tokenizer's chat template.
    
    Args:
        tokenizer: HuggingFace tokenizer
        func_str: Function code string
        exec_result: Expected execution result
        args_str: Function arguments (use "????" for unknown)
        
    Returns:
        str: Formatted chat prompt
    """
    r1_prefix = [
        {"role": "system", "content": "You are a programming expert"},
        {
            "role": "user",
            "content": get_prompt_str(
                func_str, extract_func_name(func_str), exec_result, args_str
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        r1_prefix, tokenize=False, add_generation_prompt=True
    )

def test_python_code(code_string):
    """
    Test if Python code executes successfully.
    
    Args:
        code_string: Python code to execute
        
    Returns:
        bool: True if code runs successfully, False otherwise
    """
    try:
        process = subprocess.run(
            [sys.executable, "-c", code_string],
            capture_output=True,
            text=True,
            timeout=0.5,
            check=False
        )

        if process.returncode == 0:
            return True

        if "AssertionError" in process.stderr:
            return False
    except Exception:
        return False

    return False
