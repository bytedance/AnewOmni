#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import json
import os
import time
import uuid
import contextlib
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

from .core import GenerationResult, resolve_project_path, sdf_to_svg, sdf_to_svg_pages
from .repl import PromptREPL

import ray  # type: ignore


STATIC_DIR = os.path.join(os.path.dirname(__file__), "web_static")
DEFAULT_DEMO_ID = "demo:molecule"


def _demo_code(demo_name: str) -> str:
    ckpt = resolve_project_path("checkpoints/model.ckpt")
    pdb = resolve_project_path("demo/data/8u4r_chothia.pdb")

    if demo_name == "molecule":
        return "\n".join(
            [
                "graph = MoleculePrompt()",
                'graph.add_fragment("c1ccccc1", name="ring")',
                'graph.add_fragment("C", name="methyl")',
                'graph.add_bond("ring:c0", "methyl:c0")',
                "graph.allow_growth(3, 6)",
                "graph.set_context(",
                f'    pdb_path="{pdb}",',
                '    tgt_chains="R",',
                '    lig_chains="HL",',
                f'    checkpoint="{ckpt}",',
                "    gpu=0,",
                ")",
            ]
        )
    if demo_name == "peptide":
        return "\n".join(
            [
                "pep = PeptidePrompt(length=12)",
                'pep.add_motif("RGD")',
                # random nsAA insertion (no positions)
                'pep.add_noncanonical("NC1=CC=C(C=C1)C=C(C#N)C2=CC=CC(C(C(=O)*)N*)=C2", count=1)',
                "pep.set_context(",
                f'    pdb_path="{pdb}",',
                '    tgt_chains="R",',
                '    lig_chains="HL",',
                f'    checkpoint="{ckpt}",',
                "    gpu=0,",
                ")",
            ]
        )
    if demo_name == "cyclic_peptide":
        return "\n".join(
            [
                "pep = PeptidePrompt(length=12)",
                'pep.add_motif("RGD")',
                # choose one: head-tail or disulfide
                'pep.cyclize(mode="disulfide")',
                "pep.set_context(",
                f'    pdb_path="{pdb}",',
                '    tgt_chains="R",',
                '    lig_chains="HL",',
                f'    checkpoint="{ckpt}",',
                "    gpu=0,",
                ")",
            ]
        )
    # antibody
    return "\n".join(
        [
            f'ab = AntibodyPrompt(framework_path="{pdb}", cdrs=["HCDR3", "LCDR3"])',
            'ab.set_length("HCDR3", 12)',
            'ab.set_length("LCDR3", 9)',
            'ab.add_motif("HCDR3", "YYG", positions=[2, 3, 4])',
            "ab.set_context(",
            f'    pdb_path="{pdb}",',
            '    tgt_chains="R",',
            '    lig_chains="HL",',
            f'    checkpoint="{ckpt}",',
            "    gpu=0,",
            ")",
        ]
    )


def _read_json_body(handler: SimpleHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


@dataclass
class TaskRecord:
    id: str
    code: str
    save_dir: str
    submitted_at: float
    status: str  # queued | running | completed | failed
    error: Optional[str] = None
    prompt_snapshot: Optional[Dict[str, Any]] = None
    state_text_snapshot: Optional[str] = None
    result_payload: Optional[Dict[str, Any]] = None
    ray_ref: Any = None
    log_path: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "save_dir": self.save_dir,
            "submitted_at": self.submitted_at,
            "status": self.status,
            "error": self.error,
        }


def _build_result_payload_from_save_dir(save_dir: str) -> Dict[str, Any]:
    raw_records_path = os.path.join(save_dir, "results.jsonl")
    filtered_records_path = os.path.join(save_dir, "filtered_results.jsonl")
    records = []
    records_path = raw_records_path
    if os.path.exists(filtered_records_path):
        with open(filtered_records_path, "r") as fin:
            filtered_records = [json.loads(line) for line in fin.readlines()]
        if filtered_records:
            records = filtered_records
            records_path = filtered_records_path
    if not records and os.path.exists(raw_records_path):
        with open(raw_records_path, "r") as fin:
            records = [json.loads(line) for line in fin.readlines()]

    first_record = records[0] if records else None
    svg = None
    svg_pages: List[str] = []
    sdf_path = None
    cif_path = None
    if first_record is not None:
        result_prefix = os.path.join(save_dir, first_record["id"], str(first_record["n"]))
        sdf_path = result_prefix + ".sdf"
        cif_path = result_prefix + ".cif"
    if sdf_path is not None and os.path.exists(sdf_path):
        try:
            svg = sdf_to_svg(sdf_path)
            svg_pages = sdf_to_svg_pages(sdf_path)
        except Exception:
            svg = None
            svg_pages = []

    return {
        "save_dir": save_dir,
        "records_path": records_path,
        "raw_records_path": raw_records_path if os.path.exists(raw_records_path) else None,
        "filtered_records_path": filtered_records_path if os.path.exists(filtered_records_path) else None,
        "display_source": "filtered" if records_path == filtered_records_path and records else "raw",
        "used_fallback": bool(first_record.get("selection_reason") == "best_confidence_fallback") if first_record else False,
        "selection_reason": first_record.get("selection_reason") if first_record else None,
        "records": records,
        "svg": svg,
        "svg_pages": svg_pages,
        "sdf_path": sdf_path,
        "cif_path": cif_path,
        "cif_url": f"/api/file?path={quote(cif_path)}" if cif_path and os.path.exists(cif_path) else None,
    }


def _ensure_ray_initialized():
    if ray is None:
        raise RuntimeError("Ray is not available in this environment")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)


def _available_gpus() -> int:
    if ray is None or not ray.is_initialized():
        return 0
    resources = ray.cluster_resources()
    return int(resources.get("GPU", 0))


def _make_save_dir(save_dir: Optional[str]) -> str:
    if save_dir is not None and save_dir.strip():
        return save_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.getcwd(), "outputs", f"ui_web_run_{timestamp}")

def _tail_file(path: str, max_lines: int = 200) -> str:
    if not path or not os.path.exists(path):
        return ""
    # Simple tail implementation; file size is small for UI logs.
    with open(path, "r", errors="ignore") as fin:
        lines = fin.readlines()
    return "".join(lines[-max_lines:])

def _task_run_impl(code: str, save_dir: str, use_gpu: bool) -> Dict[str, Any]:
    # This function is executed inside a Ray worker.
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "run.log")
    log_f = open(log_path, "a", buffering=1)
    log_f.write(f"[ui] task started, use_gpu={use_gpu}\n")
    repl = PromptREPL()
    with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        repl.eval(code)
    prompt = repl.get_active_prompt()
    if prompt is None:
        log_f.write("[ui] no active prompt created by code\n")
        log_f.close()
        raise ValueError("No active prompt was created by the submitted code")

    prompt_snapshot = prompt.to_visual_payload()
    state_text = repl.render_state()

    # When running inside Ray, CUDA_VISIBLE_DEVICES will be set for GPU tasks.
    # Using gpu=0 maps to the first visible GPU.
    gpu_value = 0 if use_gpu else -1
    log_f.write("[ui] starting generation\n")
    with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        result = prompt.run_generation(save_dir, gpu=gpu_value, _ui_log=log_f.write)
    log_f.write("[ui] generation finished\n")
    result_payload = _build_result_payload_from_save_dir(result.save_dir)
    result_payload["summary"] = result.summary()
    log_f.close()

    return {
        "code": code,
        "save_dir": result.save_dir,
        "log_path": log_path,
        "prompt_snapshot": prompt_snapshot,
        "state_text_snapshot": state_text,
        "result_payload": result_payload,
    }


_ray_task_gpu = None
_ray_task_cpu = None
if ray is not None:
    _ray_task_gpu = ray.remote(num_gpus=0.6)(_task_run_impl)
    _ray_task_cpu = ray.remote(num_gpus=0)(_task_run_impl)


@dataclass
class WebState:
    repl: PromptREPL
    latest_result: Optional[Dict[str, Any]] = None
    tasks: Dict[str, TaskRecord] = None
    selected_task_id: Optional[str] = None

    def _current_prompt_payload(self) -> Dict[str, Any]:
        prompt = self.repl.get_active_prompt()
        if prompt is None:
            return {"kind": "empty", "text": "No active prompt", "svg": None, "fragment_mappings": []}
        return prompt.to_visual_payload()

    def _result_payload_from_result(self, result: GenerationResult) -> Dict[str, Any]:
        payload = _build_result_payload_from_save_dir(result.save_dir)
        payload["summary"] = result.summary()
        return payload

    def _sync_task_states(self):
        if self.tasks is None:
            self.tasks = {}
        if ray is None or not ray.is_initialized():
            return
        for task in self.tasks.values():
            if task.status in {"demo"}:
                continue
            if task.status not in {"queued", "running"}:
                continue
            if task.ray_ref is None:
                continue
            # First poll moves queued -> running. This is an approximation.
            if task.status == "queued":
                task.status = "running"
            ready, _ = ray.wait([task.ray_ref], timeout=0)
            if not ready:
                continue
            try:
                out = ray.get(task.ray_ref)
                task.status = "completed"
                task.prompt_snapshot = out.get("prompt_snapshot")
                task.state_text_snapshot = out.get("state_text_snapshot")
                task.result_payload = out.get("result_payload")
                task.log_path = out.get("log_path", task.log_path)
            except Exception as exc:
                task.status = "failed"
                task.error = str(exc)

    def serialize(self) -> Dict[str, Any]:
        self._sync_task_states()
        tasks: List[Dict[str, Any]] = []
        if self.tasks:
            tasks = sorted((t.summary() for t in self.tasks.values()), key=lambda x: x["submitted_at"], reverse=True)

        selected_task = self.tasks.get(self.selected_task_id) if self.tasks and self.selected_task_id else None
        view_prompt = self._current_prompt_payload()
        view_state_text = self.repl.render_state()
        view_latest_result = self.latest_result
        view_code = None
        view_task_status = None
        view_task_error = None
        view_log = ""

        if selected_task is not None:
            view_code = selected_task.code
            view_task_status = selected_task.status
            view_task_error = selected_task.error
            view_log = _tail_file(selected_task.log_path or "")

            if selected_task.status == "demo":
                # Demo is treated as "live editable": selecting a demo should not overwrite
                # the current prompt with an old snapshot.
                pass
            else:
                if selected_task.prompt_snapshot is not None:
                    view_prompt = selected_task.prompt_snapshot
                if selected_task.state_text_snapshot is not None:
                    view_state_text = selected_task.state_text_snapshot
                # While running, show a placeholder in latest_result.
                if selected_task.status in {"queued", "running"}:
                    view_latest_result = {
                        "summary": f"Task {selected_task.id} is {selected_task.status}",
                        "save_dir": selected_task.save_dir,
                        "records_path": None,
                        "records": [],
                        "svg": None,
                        "svg_pages": [],
                        "sdf_path": None,
                        "cif_path": None,
                        "cif_url": None,
                        "task_status": selected_task.status,
                    }
                elif selected_task.status == "failed":
                    view_latest_result = {
                        "summary": f"Task {selected_task.id} failed: {selected_task.error}",
                        "save_dir": selected_task.save_dir,
                        "records_path": None,
                        "records": [],
                        "svg": None,
                        "svg_pages": [],
                        "sdf_path": None,
                        "cif_path": None,
                        "cif_url": None,
                        "task_status": selected_task.status,
                    }
                elif selected_task.status == "completed":
                    view_latest_result = selected_task.result_payload

        return {
            "prompt": view_prompt,
            "state_text": view_state_text,
            "latest_result": view_latest_result,
            "tasks": tasks,
            "selected_task_id": self.selected_task_id,
            "view": {
                "code": view_code,
                "task_status": view_task_status,
                "task_error": view_task_error,
                "log": view_log,
            },
        }

    def eval_code(self, code: str) -> Dict[str, Any]:
        result = self.repl.eval(code)
        # Evaluating code is a "live editing" action. If a demo is selected, keep it selected.
        # Otherwise, switch back to Live view so users see updated prompt instead of a task snapshot.
        if not (self.selected_task_id or "").startswith("demo:"):
            self.selected_task_id = None
        return {
            "output": result.output,
            "state": result.state,
            "should_exit": result.should_exit,
            "app_state": self.serialize(),
        }

    def reset(self) -> Dict[str, Any]:
        result = self.repl.reset()
        self.latest_result = None
        self.selected_task_id = None
        return {
            "output": result.output,
            "app_state": self.serialize(),
        }

    def run_generation(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        prompt = self.repl.get_active_prompt()
        if prompt is None:
            raise ValueError("No active prompt")
        if save_dir is None or save_dir.strip() == "":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(os.getcwd(), "outputs", f"ui_web_run_{timestamp}")
        result = prompt.run_generation(save_dir)
        self.latest_result = self._result_payload_from_result(result)
        return {
            "output": result.summary(),
            "app_state": self.serialize(),
        }

    def submit_task(self, code: str, save_dir: Optional[str]) -> Dict[str, Any]:
        _ensure_ray_initialized()
        if self.tasks is None:
            self.tasks = {}

        task_id = str(uuid.uuid4())[:8]
        save_dir_resolved = _make_save_dir(save_dir)

        use_gpu = _available_gpus() >= 1
        # Run inside Ray so it can schedule based on available GPU resources.
        if use_gpu:
            ray_ref = _ray_task_gpu.remote(code, save_dir_resolved, True)
        else:
            ray_ref = _ray_task_cpu.remote(code, save_dir_resolved, False)

        record = TaskRecord(
            id=task_id,
            code=code,
            save_dir=save_dir_resolved,
            submitted_at=time.time(),
            status="queued",
            prompt_snapshot=self._current_prompt_payload(),
            state_text_snapshot=self.repl.render_state(),
            ray_ref=ray_ref,
            log_path=os.path.join(save_dir_resolved, "run.log"),
        )
        self.tasks[task_id] = record
        self.selected_task_id = task_id

        return {"task_id": task_id, "status": record.status, "app_state": self.serialize()}

    def select_task(self, task_id: Optional[str]) -> Dict[str, Any]:
        if task_id is None or task_id == "":
            self.selected_task_id = None
            return {"ok": True, "app_state": self.serialize()}
        if self.tasks is None or task_id not in self.tasks:
            raise ValueError(f"Unknown task id: {task_id}")
        self.selected_task_id = task_id
        return {"ok": True, "app_state": self.serialize()}

    def list_tasks(self) -> Dict[str, Any]:
        return {"tasks": self.serialize().get("tasks", [])}


class UIRequestHandler(SimpleHTTPRequestHandler):
    server_version = "AnewUI/0.1"

    def __init__(self, *args, app_state: WebState, **kwargs):
        self.app_state = app_state
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _write_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # Always disable caching for the UI so browser refreshes pick up the latest
        # `index.html` / `app.js` / `styles.css` during rapid local iteration.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path == "/api/state":
            self._write_json(self.app_state.serialize())
            return
        if self.path == "/api/tasks":
            self._write_json(self.app_state.list_tasks())
            return
        if self.path.startswith("/api/file"):
            self._serve_file()
            return
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/eval":
                body = _read_json_body(self)
                self._write_json(self.app_state.eval_code(body.get("code", "")))
                return
            if self.path == "/api/reset":
                self._write_json(self.app_state.reset())
                return
            if self.path == "/api/run_generation":
                body = _read_json_body(self)
                self._write_json(self.app_state.run_generation(body.get("save_dir", None)))
                return
            if self.path == "/api/tasks/submit":
                body = _read_json_body(self)
                self._write_json(self.app_state.submit_task(body.get("code", ""), body.get("save_dir", None)))
                return
            if self.path == "/api/tasks/select":
                body = _read_json_body(self)
                self._write_json(self.app_state.select_task(body.get("task_id", None)))
                return
            self._write_json({"error": f"Unknown endpoint: {self.path}"}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _serve_file(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        file_path = query.get("path", [None])[0]
        if file_path is None:
            self._write_json({"error": "Missing path parameter"}, status=HTTPStatus.BAD_REQUEST)
            return
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            self._write_json({"error": f"File not found: {file_path}"}, status=HTTPStatus.NOT_FOUND)
            return
        ext = os.path.splitext(file_path)[1].lower()
        content_type = {
            ".cif": "chemical/x-cif",
            ".pdb": "chemical/x-pdb",
            ".sdf": "chemical/x-mdl-sdfile",
            ".json": "application/json",
        }.get(ext, "application/octet-stream")
        with open(file_path, "rb") as fin:
            data = fin.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(port: int = 8765) -> int:
    # Allow overriding the port without adding extra CLI dependencies.
    port = int(os.environ.get("ANEW_UI_PORT", str(port)))
    if ray is not None:
        # Initialize Ray early so task submission is fast.
        _ensure_ray_initialized()
    demo_tasks: Dict[str, TaskRecord] = {
        "demo:molecule": TaskRecord(id="demo:molecule", code=_demo_code("molecule"), save_dir="", submitted_at=0.0, status="demo"),
        "demo:peptide": TaskRecord(id="demo:peptide", code=_demo_code("peptide"), save_dir="", submitted_at=0.0, status="demo"),
        "demo:cyclic_peptide": TaskRecord(id="demo:cyclic_peptide", code=_demo_code("cyclic_peptide"), save_dir="", submitted_at=0.0, status="demo"),
        "demo:antibody": TaskRecord(id="demo:antibody", code=_demo_code("antibody"), save_dir="", submitted_at=0.0, status="demo"),
    }
    app_state = WebState(repl=PromptREPL(), tasks=demo_tasks, selected_task_id=DEFAULT_DEMO_ID)
    handler = partial(UIRequestHandler, app_state=app_state)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"UI server listening on http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
