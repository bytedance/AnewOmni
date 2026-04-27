#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import (
    AbnormalConfidenceFilter,
    AntibodyPrompt,
    ChainBreakFilter,
    ChiralCentersFilter,
    ConfidenceThresholdFilter,
    LTypeAAFilter,
    MolSMARTSFilter,
    MolWeightFilter,
    MoleculePrompt,
    PeptidePrompt,
    PhysicalValidityFilter,
    PromptProgram,
    RotatableBondsFilter,
    SimpleClashFilter,
    SimpleGeometryFilter,
)


@dataclass
class EvalResult:
    output: str
    state: str = ""
    should_exit: bool = False


class _SafetyVisitor(ast.NodeVisitor):
    _blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.AsyncWith,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.Raise,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Lambda,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )

    _blocked_calls = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "help",
    }

    def generic_visit(self, node: ast.AST):
        if isinstance(node, self._blocked_nodes):
            raise ValueError(f"Blocked syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is blocked")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self._blocked_calls:
            raise ValueError(f"Blocked call: {node.func.id}")
        self.generic_visit(node)


class PromptREPL:
    def __init__(self):
        self._active_prompt_key = "__active_prompt__"
        self.globals = self._make_globals()

    def _make_globals(self) -> Dict[str, Any]:
        safe_builtins = {
            "abs": abs,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }
        return {
            "__builtins__": safe_builtins,
            self._active_prompt_key: None,
            "AbnormalConfidenceFilter": AbnormalConfidenceFilter,
            "AntibodyPrompt": AntibodyPrompt,
            "ChainBreakFilter": ChainBreakFilter,
            "ChiralCentersFilter": ChiralCentersFilter,
            "ConfidenceThresholdFilter": ConfidenceThresholdFilter,
            "LTypeAAFilter": LTypeAAFilter,
            "MolSMARTSFilter": MolSMARTSFilter,
            "MolWeightFilter": MolWeightFilter,
            "MoleculePrompt": MoleculePrompt,
            "PeptidePrompt": PeptidePrompt,
            "PhysicalValidityFilter": PhysicalValidityFilter,
            "RotatableBondsFilter": RotatableBondsFilter,
            "SimpleClashFilter": SimpleClashFilter,
            "SimpleGeometryFilter": SimpleGeometryFilter,
        }

    def reset(self) -> EvalResult:
        self.globals = self._make_globals()
        return EvalResult(output="Session reset")

    def get_active_prompt(self) -> Optional[PromptProgram]:
        active = self.globals.get(self._active_prompt_key, None)
        if isinstance(active, PromptProgram):
            return active
        for _, value in reversed(list(self.globals.items())):
            if isinstance(value, PromptProgram):
                return value
        return None

    def render_state(self) -> str:
        prompt = self.get_active_prompt()
        if prompt is None:
            return "No active prompt"
        return prompt.inspect()

    def eval(self, source: str) -> EvalResult:
        stripped = source.strip()
        if not stripped:
            return EvalResult(output="")
        if stripped.startswith(":"):
            return self._handle_command(stripped)
        tree = ast.parse(source, mode="exec")
        _SafetyVisitor().visit(tree)
        output = self._execute_tree(tree)
        state = self.render_state()
        return EvalResult(output=output, state=state)

    def _execute_tree(self, tree: ast.Module) -> str:
        if not tree.body:
            return ""
        if isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            if prefix.body:
                exec(compile(prefix, "<prompt-repl>", "exec"), self.globals, self.globals)
            expr = ast.Expression(tree.body[-1].value)
            value = eval(compile(expr, "<prompt-repl>", "eval"), self.globals, self.globals)
            if isinstance(value, PromptProgram):
                # Track the last prompt-producing expression explicitly; relying on dict insertion
                # order breaks when users reassign existing variable names.
                self.globals[self._active_prompt_key] = value
            return "" if value is None else str(value)
        exec(compile(tree, "<prompt-repl>", "exec"), self.globals, self.globals)
        # Heuristic: if the code ends with an assignment, consider the assigned prompt active.
        for node in reversed(tree.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value = self.globals.get(target.id, None)
                        if isinstance(value, PromptProgram):
                            self.globals[self._active_prompt_key] = value
                            return "OK"
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                value = self.globals.get(node.target.id, None)
                if isinstance(value, PromptProgram):
                    self.globals[self._active_prompt_key] = value
                    return "OK"
        return "OK"

    def _handle_command(self, command: str) -> EvalResult:
        if command == ":help":
            return EvalResult(output=self._help_text())
        if command == ":state":
            return EvalResult(output=self.render_state())
        if command == ":labels":
            prompt = self.get_active_prompt()
            if prompt is None:
                return EvalResult(output="No active prompt")
            labels = getattr(prompt, "labels", None)
            if labels is None:
                return EvalResult(output="The active prompt does not expose labels()")
            return EvalResult(output="\n".join(labels()))
        if command == ":compile":
            prompt = self.get_active_prompt()
            if prompt is None:
                return EvalResult(output="No active prompt")
            return EvalResult(output=prompt.compile().summary())
        if command.startswith(":run "):
            prompt = self.get_active_prompt()
            if prompt is None:
                return EvalResult(output="No active prompt")
            save_dir = command.split(" ", 1)[1].strip()
            result = prompt.run_generation(save_dir)
            return EvalResult(output=result.summary(), state=self.render_state())
        if command == ":reset":
            return self.reset()
        if command == ":quit":
            return EvalResult(output="Bye", should_exit=True)
        raise ValueError(f"Unknown command: {command}")

    def _help_text(self) -> str:
        return "\n".join(
            [
                "Prompt REPL commands:",
                ":help        Show this message",
                ":state       Show the current prompt state",
                ":labels      Show available labels if the active prompt supports them",
                ":compile     Show the compiled prompt summary",
                ":run <dir>   Run generation with the active prompt",
                ":reset       Reset the REPL session",
                ":quit        Exit the REPL",
            ]
        )


def _run_script(path: str) -> int:
    repl = PromptREPL()
    with open(path, "r") as fin:
        source = fin.read()
    result = repl.eval(source)
    if result.output:
        print(result.output)
    if result.state:
        print("\nCurrent Prompt")
        print(result.state)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return _run_script(argv[0])

    repl = PromptREPL()
    print("Prompt REPL ready. Type :help for commands.")
    while True:
        try:
            line = input(">>> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        try:
            result = repl.eval(line)
            if result.output:
                print(result.output)
            if result.state and result.output != result.state:
                print("\nCurrent Prompt")
                print(result.state)
            if result.should_exit:
                return 0
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
