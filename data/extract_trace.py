"""
Extract Trace - Step 9 of ExecVerify Data Pipeline

Generates execution traces for candidate samples by executing the code
and recording line-by-line execution with variable states.

Input: candidates_io_for_multi_task_with_cot.json
Output: io_dataset_for_mutiple_tasks_with_traces.json
"""

import sys
import ast
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from utils import load_config, setup_logging, load_json, save_json


def _safe_repr(obj, maxlen: int = 200) -> str:
    """Safely represent object as string."""
    try:
        s = repr(obj)
    except Exception:
        s = f"<unrepr {type(obj).__name__}>"
    if len(s) > maxlen:
        s = s[:maxlen] + "...<truncated>"
    return s


def run_and_trace_script(source_code: str, filename_tag: str = "__user_src__"):
    """
    Execute source code and trace all user-defined functions.
    
    Args:
        source_code: Python source code to execute
        filename_tag: Tag for identifying the source file
        
    Returns:
        Tuple of (result, trace_events, success)
    """
    trace_events = []
    success = True
    result = None
    
    def _tracer(frame, event, arg):
        # Only trace current source file frames, ignore library functions
        if frame.f_code.co_filename != filename_tag:
            return _tracer
        
        # Ignore module-level events
        if frame.f_code.co_name == "<module>":
            return _tracer
        
        rec = {
            "event": event,
            "func": frame.f_code.co_name,
            "line_no": frame.f_lineno,
            "frame_id": id(frame)  # Add frame ID for scope tracking
        }
        
        if event in ("call", "line"):
            try:
                rec["locals"] = {k: _safe_repr(v) for k, v in frame.f_locals.items()}
            except Exception:
                rec["locals"] = {"<error>": "<failed to capture locals>"}
        elif event == "return":
            rec["return"] = _safe_repr(arg)
        elif event == "exception":
            exc_type, exc_value, _tb = arg
            rec["exception"] = {
                "type": getattr(exc_type, "__name__", str(exc_type)),
                "value": _safe_repr(exc_value),
            }
        
        trace_events.append(rec)
        return _tracer
    
    # Rewrite last expression as assignment if needed
    try:
        tree = ast.parse(source_code, filename_tag, mode="exec")
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1]
            assign = ast.Assign(
                targets=[ast.Name(id="__RESULT__", ctx=ast.Store())],
                value=last_expr.value,
            )
            ast.copy_location(assign, last_expr)
            tree.body[-1] = assign
            ast.fix_missing_locations(tree)
            code_obj = compile(tree, filename_tag, "exec")
        else:
            code_obj = compile(source_code, filename_tag, "exec")
    except Exception:
        code_obj = compile(source_code, filename_tag, "exec")
    
    module_ns = {}
    sys.settrace(_tracer)
    try:
        exec(code_obj, module_ns, module_ns)
        result = module_ns.get("__RESULT__", None)
    except BaseException as e:
        success = False
        trace_events.append({
            "event": "unhandled_exception",
            "func": "<module>",
            "exception": {"type": type(e).__name__, "value": _safe_repr(e)},
        })
        result = False
    finally:
        sys.settrace(None)
    
    trace_events.append({"event": "program_end", "success": success})
    return result, trace_events, success


def extract_func_name(func_str: str) -> str:
    """Extract function name from function definition using tree-sitter."""
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
    
    if func_names:
        return func_names[0].text.decode("utf8")
    return None


def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Starting execution trace extraction...")
    
    # Load candidates
    input_file = config['output_files']['candidates_io_multi_task']
    candidates = load_json(input_file, logger)
    
    # Generate traces
    logger.info(f"Generating traces for {len(candidates)} samples...")
    io_dataset_with_traces = []
    
    for i, test_case in enumerate(candidates):
        if (i + 1) % 1000 == 0:
            logger.info(
                f"Progress: {i+1}/{len(candidates)}, "
                f"successful: {len(io_dataset_with_traces)}"
            )
        
        try:
            func_str = test_case["func_str"]
            func_name = extract_func_name(func_str)
            if func_name is None:
                continue
            
            func_args = test_case["func_args"]
            test_case["func_name"] = func_name
            
            code_to_run = f"""{func_str}

{func_name}({func_args})
"""
            
            result, trace_events, success = run_and_trace_script(code_to_run)
            
            if not success:
                continue
            
            test_case["code_to_run"] = code_to_run
            io_dataset_with_traces.append({
                "test_case": test_case,
                "trace_events": trace_events
            })
        except Exception:
            continue
    
    logger.info(f"Successfully generated {len(io_dataset_with_traces)} traces")
    
    # Save dataset with traces
    output_file = config['output_files']['io_traces']
    save_json(io_dataset_with_traces, output_file, logger)
    
    logger.info("Trace extraction complete!")


if __name__ == "__main__":
    main()
