#!/usr/bin/env python3
"""Small local server for the STS2 harness trace viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


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
        self.send_text("not found", 404)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Serve the STS2 harness trace viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", default=str(root / "logs"))
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
        },
    )
    server = FastThreadingHTTPServer((args.host, args.port), handler)
    print(f"Trace viewer: http://{args.host}:{args.port}", flush=True)
    print(f"Log dir: {handler.log_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
