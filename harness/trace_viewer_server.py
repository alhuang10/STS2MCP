#!/usr/bin/env python3
"""Small local server for the STS2 harness trace viewer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


MAX_ANNOTATION_BYTES = 5 * 1024 * 1024
READ_ERROR = object()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class FastThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def server_bind(self) -> None:
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])


class TraceViewerHandler(BaseHTTPRequestHandler):
    viewer_path: Path
    log_dir: Path
    annotation_path: Path

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: object, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status)

    def send_text(self, text: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_bytes(text.encode("utf-8"), content_type, status)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_text("not found", 404)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(path.read_bytes(), content_type)

    def log_summary(self, path: Path) -> dict[str, object]:
        line_count = 0
        last_step = None
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    line_count += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and "step" in record:
                        last_step = record["step"]
        except OSError:
            pass
        stat = path.stat()
        return {
            "name": path.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "records": line_count,
            "last_step": last_step,
        }

    def handle_logs_index(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logs = sorted(self.log_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
        self.send_json({"logs": [self.log_summary(path) for path in logs]})

    def handle_log_file(self, raw_name: str) -> None:
        name = Path(unquote(raw_name)).name
        if not name.endswith(".jsonl"):
            self.send_text("invalid log name", 400)
            return
        path = self.log_dir / name
        if not path.exists() or not path.is_file():
            self.send_text("not found", 404)
            return
        self.send_file(path)

    def annotation_info(self) -> dict[str, object]:
        line_count = 0
        if self.annotation_path.exists():
            try:
                with self.annotation_path.open("r", encoding="utf-8") as handle:
                    line_count = sum(1 for line in handle if line.strip())
            except OSError:
                line_count = 0
        return {
            "path": str(self.annotation_path),
            "records": line_count,
            "exists": self.annotation_path.exists(),
        }

    def read_json_body(self) -> object:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            self.send_text("invalid content length", 400)
            return READ_ERROR
        if length <= 0:
            self.send_text("empty request body", 400)
            return READ_ERROR
        if length > MAX_ANNOTATION_BYTES:
            self.send_text("annotation is too large", 413)
            return READ_ERROR
        try:
            data = self.rfile.read(length).decode("utf-8")
            return json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_text(f"invalid json: {exc}", 400)
            return READ_ERROR

    def handle_annotation_post(self) -> None:
        payload = self.read_json_body()
        if payload is READ_ERROR:
            return
        if not isinstance(payload, dict):
            self.send_text("annotation must be a JSON object", 400)
            return
        if payload.get("kind") != "sts2_dpo_annotation":
            self.send_text("annotation kind must be sts2_dpo_annotation", 400)
            return
        if not isinstance(payload.get("prompt"), dict) or not isinstance(payload.get("game_state"), dict):
            self.send_text("annotation must include prompt and game_state objects", 400)
            return
        if not isinstance(payload.get("chosen"), dict) or not isinstance(payload.get("rejected"), dict):
            self.send_text("annotation must include chosen and rejected objects", 400)
            return

        record = dict(payload)
        record["saved_at"] = dt.datetime.now(dt.UTC).isoformat()
        self.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        with self.annotation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        info = self.annotation_info()
        self.send_json({"status": "ok", "annotation": info})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/trace_viewer.html"}:
            self.send_file(self.viewer_path)
            return
        if path == "/api/logs":
            self.handle_logs_index()
            return
        if path.startswith("/api/logs/"):
            self.handle_log_file(path.removeprefix("/api/logs/"))
            return
        if path == "/api/dpo-annotations":
            self.send_json(self.annotation_info())
            return
        self.send_text("not found", 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/dpo-annotations":
            self.handle_annotation_post()
            return
        self.send_text("not found", 404)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Serve the STS2 harness trace viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", default=str(root / "logs"))
    parser.add_argument(
        "--annotation-path",
        default=str(root / "data" / "sts2_dpo_annotations.jsonl"),
        help="JSONL file where replay DPO annotations are appended.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    viewer_path = Path(__file__).resolve().with_name("trace_viewer.html")
    handler = type(
        "ConfiguredTraceViewerHandler",
        (TraceViewerHandler,),
        {
            "viewer_path": viewer_path,
            "log_dir": Path(args.log_dir).expanduser().resolve(),
            "annotation_path": Path(args.annotation_path).expanduser().resolve(),
        },
    )
    server = FastThreadingHTTPServer((args.host, args.port), handler)
    print(f"Trace viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Log dir: {handler.log_dir}", flush=True)
    print(f"DPO annotations: {handler.annotation_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
