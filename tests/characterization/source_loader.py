from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path


def load_function(
    path: Path,
    function_name: str,
    globals_dict: dict[str, object],
) -> Callable[..., object]:
    """Load one function definition without executing its source module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    if function is None:
        raise LookupError(f"Function {function_name!r} not found in {path.name}")

    namespace = dict(globals_dict)
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    loaded = namespace[function_name]
    if not callable(loaded):
        raise TypeError(f"Loaded object {function_name!r} is not callable")
    return loaded
