"""Shared classification and duplicate protection for local Python actions."""

from __future__ import annotations

import ast
import hashlib
import json
import types
from collections.abc import Mapping
from typing import Any, Iterable


class DuplicateComputeActionError(RuntimeError):
    """Raised when identical local code is repeated with identical inputs."""


class _ExternalInputVisitor(ast.NodeVisitor):
    """Find names read before they are assigned by the current code block."""

    def __init__(self) -> None:
        self.assigned: set[str] = set()
        self.referenced: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id not in self.assigned:
            self.referenced.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.assigned.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.visit(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.visit(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name) and node.target.id not in self.assigned:
            self.referenced.add(node.target.id)
        else:
            self.visit(node.target)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self.assigned.add(node.target.id)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self.visit(node.target)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.assigned.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.assigned.add(alias.asname or alias.name)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        for statement in node.body:
            self.visit(statement)
        for statement in node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def _visit_comprehension(self, node: Any) -> None:
        outer_assigned = set(self.assigned)
        for generator in node.generators:
            self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if hasattr(node, "key"):
            self.visit(node.key)
        self.visit(node.elt if hasattr(node, "elt") else node.value)
        self.assigned = outer_assigned

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension
    visit_DictComp = _visit_comprehension

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        self.assigned.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.assigned.add(node.name)


def calls_registered_action(code: str, action_names: Iterable[str]) -> bool:
    """Return whether *code* directly calls one of the registered actions."""
    names = {str(name) for name in action_names}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    # Follow simple aliases so ``detect = ObjectLocation2D; detect(...)`` is
    # still a tool action rather than being mislabeled as local Compute.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Name) or value.id not in names:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.add(target.id)
                    changed = True
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in names
        for node in ast.walk(tree)
    )


def _jsonable_state_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_state_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_state_value(item) for item in value]
    if isinstance(value, set):
        rendered = [_jsonable_state_value(item) for item in value]
        return sorted(rendered, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return _jsonable_state_value(value.tolist())
        except Exception:
            pass
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def state_snapshot(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Capture persistent, serializable user state while excluding runtime helpers."""
    snapshot: dict[str, Any] = {}
    for key, value in (state or {}).items():
        if str(key).startswith("__") or key == "print_outputs":
            continue
        if callable(value) or isinstance(value, types.ModuleType):
            continue
        snapshot[str(key)] = _jsonable_state_value(value)
    return snapshot


def state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Return the persisted variables added or changed by one code action."""
    missing = object()
    changes: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        old = before.get(key, missing)
        new = after.get(key, missing)
        if old == new:
            continue
        changes[key] = {
            "before": None if old is missing else old,
            "after": None if new is missing else new,
        }
    return changes


def compute_action_signature(code: str, state: Mapping[str, Any] | None = None) -> str:
    """Hash canonical code together with values of referenced persistent inputs."""
    try:
        tree = ast.parse(code)
        canonical_code = ast.dump(tree, annotate_fields=True, include_attributes=False)
        visitor = _ExternalInputVisitor()
        visitor.visit(tree)
        referenced_names = visitor.referenced
    except SyntaxError:
        canonical_code = " ".join(code.split())
        referenced_names = set()

    snapshot = state_snapshot(state)
    inputs = {name: snapshot[name] for name in sorted(referenced_names) if name in snapshot}
    payload = json.dumps(
        {"code": canonical_code, "inputs": inputs},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ComputeActionGuard:
    """Episode-local duplicate guard shared by the traditional inference agents."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def reset(self) -> None:
        self._seen.clear()

    def check(self, code: str, state: Mapping[str, Any] | None = None) -> str:
        signature = compute_action_signature(code, state)
        if signature in self._seen:
            raise DuplicateComputeActionError(
                "Identical Compute action rejected. Reuse its prior Observation or change its inputs."
            )
        return signature

    def record(self, signature: str) -> None:
        self._seen.add(signature)
