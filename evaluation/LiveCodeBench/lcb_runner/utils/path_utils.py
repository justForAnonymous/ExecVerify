import pathlib
from typing import Union

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: Union[str, pathlib.Path], is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def _normalize_model_repr(model_repr: Union[str, LanguageModel]) -> str:
    if isinstance(model_repr, LanguageModel):
        return model_repr.model_repr
    return model_repr


def _get_output_base_dir(model_repr: Union[str, LanguageModel], args) -> pathlib.Path:
    if getattr(args, "output_dir", None):
        return pathlib.Path(args.output_dir)
    return pathlib.Path("output") / _normalize_model_repr(model_repr)


def get_cache_path(model_repr: str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    normalized_repr = _normalize_model_repr(model_repr)
    path = pathlib.Path("cache") / normalized_repr / f"{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return str(path)


def get_output_path(model_repr: Union[str, LanguageModel], args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    base_dir = _get_output_base_dir(model_repr, args)
    file_name = f"{scenario}_{n}_{temperature}{cot_suffix}.json"
    path = base_dir / file_name
    ensure_dir(path)
    return str(path)


def get_eval_all_output_path(model_repr: Union[str, LanguageModel], args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    base_dir = _get_output_base_dir(model_repr, args)
    file_name = f"{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"
    path = base_dir / file_name
    ensure_dir(path)
    return str(path)
