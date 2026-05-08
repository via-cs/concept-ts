"""
Safely execute LLM-generated Python filter functions.

Each prototype's "function" field is a Python function with this contract:

    def select_segments(segments: list[dict]) -> list[int]:
        # returns indices into `segments` of segments matching the prototype

The function runs in a restricted globals dict that exposes only standard
arithmetic / iteration built-ins — no imports, no I/O, no attribute access
to system modules. Errors are caught and reported alongside the empty result.
"""

from __future__ import annotations

import traceback


# ---------------------------------------------------------------------------
# Restricted globals: only safe builtins, no imports, no exec, no I/O
# ---------------------------------------------------------------------------

_SAFE_BUILTINS = {
    name: getattr(__builtins__, name) if hasattr(__builtins__, name)
    else __builtins__[name]
    for name in [
        # types
        "bool", "int", "float", "str", "list", "dict", "set", "tuple",
        # iteration / collection
        "len", "range", "enumerate", "zip", "map", "filter", "sorted",
        "reversed", "any", "all", "sum", "min", "max",
        # numeric
        "abs", "round", "pow", "divmod",
        # introspection (safe forms)
        "isinstance",
        # printing — useful for the LLM's debugging, harmless
        "print",
    ]
    if (hasattr(__builtins__, name) if not isinstance(__builtins__, dict)
        else name in __builtins__)
}


def execute_select_function(function_code: str, segments: list,
                             function_name: str = "select_segments") -> dict:
    """Compile and run an LLM-supplied filter function, return matching indices.

    Returns
    -------
    dict with keys:
        matched_indices : list[int] | None  (None if anything went wrong)
        error           : str | None        (traceback string, or None on success)
    """
    if not function_code or not function_code.strip():
        return {"matched_indices": None, "error": "empty function code"}

    safe_globals = {"__builtins__": _SAFE_BUILTINS}
    safe_locals: dict = {}

    try:
        compiled = compile(function_code, filename="<llm_proposed>", mode="exec")
        exec(compiled, safe_globals, safe_locals)
    except Exception as e:
        return {"matched_indices": None,
                "error": f"compile/exec error: {type(e).__name__}: {e}\n"
                         f"{traceback.format_exc()}"}

    fn = safe_locals.get(function_name)
    if fn is None:
        # Fall back: any callable defined in the snippet
        callables = [v for v in safe_locals.values() if callable(v)]
        if not callables:
            return {"matched_indices": None,
                    "error": f"no function named '{function_name}' "
                             "(and no other callable) found in code"}
        fn = callables[0]

    try:
        result = fn(segments)
    except Exception as e:
        return {"matched_indices": None,
                "error": f"runtime error inside function: {type(e).__name__}: {e}\n"
                         f"{traceback.format_exc()}"}

    if not isinstance(result, list):
        return {"matched_indices": None,
                "error": f"function returned {type(result).__name__}, expected list"}

    valid = []
    for x in result:
        if not isinstance(x, int):
            return {"matched_indices": None,
                    "error": f"non-integer index in result: {x!r} ({type(x).__name__})"}
        if x < 0 or x >= len(segments):
            return {"matched_indices": None,
                    "error": f"out-of-range index in result: {x} "
                             f"(segments has {len(segments)} entries)"}
        valid.append(x)

    # Remove duplicates while preserving order
    seen = set()
    deduped = []
    for x in valid:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return {"matched_indices": deduped, "error": None}
