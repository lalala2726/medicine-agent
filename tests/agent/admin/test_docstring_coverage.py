from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ADMIN_ROOT = PROJECT_ROOT / "app" / "agent" / "admin"


def _iter_target_functions(py_file: Path) -> list[tuple[str, int, str]]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            functions.append((node.name, node.lineno, doc))
    return functions


def test_admin_functions_have_docstring_with_args_and_returns():
    missing_doc: list[str] = []
    missing_args: list[str] = []
    missing_returns: list[str] = []

    for py_file in sorted(ADMIN_ROOT.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        for function_name, lineno, docstring in _iter_target_functions(py_file):
            location = f"{py_file}:{lineno}:{function_name}"
            if not docstring.strip():
                missing_doc.append(location)
                continue
            if "Args:" not in docstring:
                missing_args.append(location)
            if "Returns:" not in docstring:
                missing_returns.append(location)

    assert not missing_doc, "以下函数缺少 docstring:\n" + "\n".join(missing_doc)
    assert not missing_args, "以下函数 docstring 缺少 Args:\n" + "\n".join(missing_args)
    assert not missing_returns, "以下函数 docstring 缺少 Returns:\n" + "\n".join(missing_returns)
