# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reject incomplete or changed HTTP evidence before comparing configurations."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks import benchmark_response_equivalence as bench


@pytest.fixture
def response():
    return {
        "id": "response-id",
        "created": 1,
        "model": "test-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 128, "total_tokens": 138},
        "choices": [
            {
                "index": 0,
                "text": "ok",
                "finish_reason": "length",
                "logprobs": {
                    "tokens": ["x"] * 128,
                    "token_logprobs": [-0.1] * 128,
                    "top_logprobs": [{"x": -0.1}] * 128,
                    "text_offset": list(range(128)),
                },
            }
        ],
    }


@pytest.fixture
def server(response):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        worker.join()
        http.server_close()


@pytest.fixture
def cohorts(tmp_path, server):
    for arm in ("A", "P"):
        assert (
            bench.main(
                [
                    "run",
                    "--url",
                    server,
                    "--model",
                    "test-model",
                    "--label",
                    arm,
                    "--output",
                    str(tmp_path / arm),
                    "--workload",
                    "completions",
                    "--warmup",
                    "1",
                    "--requests",
                    "2",
                    "--concurrency",
                    "1",
                ]
            )
            == 0
        )
    return tmp_path


def compare(root):
    return bench.main(
        [
            "compare",
            "--baseline",
            str(root / "A"),
            "--candidate",
            str(root / "P"),
            "--output",
            str(root / "comparison.json"),
        ]
    )


def rewrite(root, change, *, raw_matches=True):
    path = root / "P/requests.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    if raw_matches:
        for record in records:
            record["response_body_hex"] = json.dumps(record["response"]).encode().hex()
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_complete_recording_compares_warmups_and_preserves_original_bytes(cohorts):
    rewrite(cohorts, lambda rows: rows[0]["response"].update(id="new", created=2))
    assert compare(cohorts) == 0
    report = json.loads((cohorts / "comparison.json").read_text())
    assert report["baseline_requests"] == report["candidate_requests"] == 3


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[0]["response"]["choices"][0].update(text="changed warmup"),
        lambda rows: rows[1]["response"]["choices"][0]["logprobs"][
            "token_logprobs"
        ].__setitem__(0, -0.2),
        lambda rows: rows[1]["response"]["choices"][0]["logprobs"][
            "text_offset"
        ].__setitem__(1, True),
        lambda rows: rows[0].update(status=500, success=False, errors=["HTTP 500"]),
        lambda rows: rows[1]["request"].update(seed=1),
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0]),
    ],
)
def test_changes_fail_even_when_summary_claims_success(cohorts, change):
    rewrite(cohorts, change)
    assert compare(cohorts) != 0


def test_edited_parsed_response_cannot_override_original_http_bytes(cohorts):
    rewrite(
        cohorts, lambda rows: rows[0]["response"].update(created=20), raw_matches=False
    )
    assert compare(cohorts) == 2


def test_only_generated_tool_ids_are_ignored(response):
    response["choices"][0]["message"] = {
        "tool_calls": [
            {
                "id": "dynamic",
                "type": "function",
                "function": {"name": "f", "arguments": '{"id":1}'},
            }
        ]
    }
    normalized = bench.normalize(response)
    assert "id" not in normalized["choices"][0]["message"]["tool_calls"][0]
    assert (
        normalized["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == '{"id":1}'
    )
    assert response["choices"][0]["message"]["tool_calls"][0]["id"] == "dynamic"


def test_invalid_json_body_is_retained(monkeypatch):
    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"\xffnot-json"

    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Reply())
    endpoint, payload = bench.workloads("test-model")["completions"]
    row = bench.request_once(
        "http://unused", endpoint, payload, "completions", "warmup", 0, 1, None
    )
    assert not row["success"]
    assert bytes.fromhex(row["response_body_hex"]) == b"\xffnot-json"
