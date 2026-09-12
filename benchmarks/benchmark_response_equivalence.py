# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Finite HTTP workload and strict response comparison; Python standard library only."""

import argparse
import concurrent.futures
import copy
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def workloads(model):
    common = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "seed": 0,
        "max_tokens": 128,
        "n": 1,
    }
    return {
        "completions": (
            "/v1/completions",
            {
                **common,
                "prompt": (
                    "List the positive integers in ascending order, starting with 1:"
                ),
                "logprobs": 5,
                "ignore_eos": True,
            },
        ),
        "chat-tools": (
            "/v1/chat/completions",
            {
                **common,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Use get_weather to find the current weather "
                            "in Hangzhou, in Celsius."
                        ),
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return current weather for a city.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name, e.g. 杭州.",
                                    },
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["city", "unit"],
                                "additionalProperties": False,
                            },
                            "strict": False,
                        },
                    }
                ],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    }


def validate_response(body, workload, model):
    errors = []
    if not isinstance(body, dict):
        return ["response must be a JSON object"]
    if "error" in body:
        errors.append("response contains an error field")
    if body.get("model") != model:
        errors.append("response model differs from requested model")
    usage = body.get("usage")
    if not isinstance(usage, dict) or any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        errors.append("missing or invalid token usage")
    elif usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        errors.append("token usage does not add up")
    elif workload == "completions" and usage["completion_tokens"] != 128:
        errors.append("completions did not produce the fixed 128 tokens")
    elif not 0 < usage["completion_tokens"] <= 128:
        errors.append("chat completion token count is outside 1..128")
    choices = body.get("choices")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        return errors + ["expected exactly one choice"]
    choice = choices[0]
    if choice.get("index") != 0 or choice.get("finish_reason") not in (
        "length",
        "stop",
        "tool_calls",
        "function_call",
    ):
        errors.append("invalid choice index or unfinished/error finish_reason")
    if workload == "completions":
        if not isinstance(choice.get("text"), str):
            errors.append("missing completion text")
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict) or any(
            not isinstance(logprobs.get(key), list) or len(logprobs[key]) != 128
            for key in ("tokens", "token_logprobs", "top_logprobs", "text_offset")
        ):
            errors.append("missing or incomplete per-token logprobs")
    elif not isinstance(choice.get("message"), dict):
        errors.append("missing chat message")
    return errors


def request_once(base_url, endpoint, payload, workload, phase, index, timeout, api_key):
    started = time.perf_counter()
    record = {
        "phase": phase,
        "workload": workload,
        "index": index,
        "endpoint": endpoint,
        "request": payload,
        "started_unix": time.time(),
        "status": None,
        "response": None,
        "usage": None,
        "errors": [],
    }
    headers = {
        "Content-Type": "application/json",
        "X-Request-ID": f"e2e-{phase}-{workload}-{index}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url + endpoint, json.dumps(payload, ensure_ascii=False).encode(), headers
    )
    raw = b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            record["status"] = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        record["status"] = error.code
        record["errors"].append(f"HTTP {error.code}")
        try:
            raw = error.read()
        except (OSError, http.client.HTTPException) as body_error:
            raw = getattr(body_error, "partial", b"")
            record["errors"].append(f"{type(body_error).__name__}: {body_error}")
    except (
        OSError,
        urllib.error.URLError,
        http.client.HTTPException,
        ValueError,
    ) as error:
        raw = getattr(error, "partial", b"")
        record["errors"].append(f"{type(error).__name__}: {error}")
    if record["status"] != 200 and not record["errors"]:
        record["errors"].append(f"unexpected HTTP status {record['status']}")
    record["latency_seconds"] = time.perf_counter() - started
    # Hex encoding retains the exact body, including malformed JSON or non-UTF-8 errors.
    record["response_body_hex"] = raw.hex()
    try:
        record["response"] = json.loads(raw)
        if isinstance(record["response"], dict):
            record["usage"] = record["response"].get("usage")
        record["errors"].extend(
            validate_response(record["response"], workload, payload["model"])
        )
    except (ValueError, UnicodeDecodeError) as error:
        record["errors"].append(f"invalid JSON response: {error}")
    record["success"] = not record["errors"] and record["status"] == 200
    return record


def percentile(values, percent):
    return (
        sorted(values)[max(0, math.ceil(len(values) * percent / 100) - 1)]
        if values
        else None
    )


def summarize(records, elapsed):
    successful = sum(record["success"] for record in records)
    tokens = successful_tokens = 0
    for record in records:
        usage = record["usage"]
        if (
            isinstance(usage, dict)
            and type(usage.get("completion_tokens")) is int
            and usage["completion_tokens"] >= 0
        ):
            tokens += usage["completion_tokens"]
            if record["success"]:
                successful_tokens += usage["completion_tokens"]
    latencies = [record["latency_seconds"] for record in records]
    return {
        "requests": len(records),
        "successful_requests": successful,
        "failed_requests": len(records) - successful,
        "wall_seconds": elapsed,
        "all_request_latency_p50_seconds": percentile(latencies, 50),
        "all_request_latency_p95_seconds": percentile(latencies, 95),
        "successful_requests_per_second": successful / elapsed if elapsed else 0,
        "successful_output_tokens": successful_tokens,
        "successful_output_tokens_per_second": successful_tokens / elapsed
        if elapsed
        else 0,
        "reported_output_tokens": tokens,
        "reported_output_tokens_per_second": tokens / elapsed if elapsed else 0,
    }


def run(args):
    parsed = urllib.parse.urlsplit(args.url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--url must be an HTTP(S) origin, optionally ending in /v1")
    if parsed.username or parsed.password:
        raise ValueError("use --api-key-env instead of credentials in the URL")
    base_url = args.url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if urllib.parse.urlsplit(base_url).path:
        raise ValueError("--url path must be empty or /v1")
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    selected = workloads(args.model)
    if args.workload != "both":
        selected = {args.workload: selected[args.workload]}
    config = {
        "label": args.label,
        "url": base_url,
        "model": args.model,
        "requests_per_workload": args.requests,
        "warmup_per_workload": args.warmup,
        "concurrency": args.concurrency,
        "socket_timeout_seconds": args.timeout,
        "workloads": selected,
        "started_unix": time.time(),
    }
    (destination / "run.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    )
    summary = {"label": args.label, "phases": {}, "success": True}
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if args.api_key_env and not api_key:
        raise ValueError(f"API key environment variable {args.api_key_env} is empty")
    with (destination / "requests.jsonl").open("w") as output:
        for workload, (endpoint, payload) in selected.items():
            for phase, count in (("warmup", args.warmup), ("measured", args.requests)):
                records = []
                started = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.concurrency
                ) as pool:
                    pending = [
                        pool.submit(
                            request_once,
                            base_url,
                            endpoint,
                            payload,
                            workload,
                            phase,
                            index,
                            args.timeout,
                            api_key,
                        )
                        for index in range(count)
                    ]
                    for future in concurrent.futures.as_completed(pending):
                        record = future.result()
                        records.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                elapsed = time.perf_counter() - started
                result = summarize(records, elapsed)
                summary["phases"][f"{workload}/{phase}"] = result
                summary["success"] &= result["failed_requests"] == 0
                print(
                    json.dumps({"phase": f"{workload}/{phase}", **result}), flush=True
                )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["success"] else 1


def normalize(response):
    value = copy.deepcopy(response)
    if not isinstance(value, dict):
        return value
    value.pop("id", None)
    value.pop("created", None)
    choices = value.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            tool_calls = (
                message.get("tool_calls") if isinstance(message, dict) else None
            )
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_call.pop("id", None)
    return value


def canonical(value):
    # Preserve JSON types as well as values (Python treats True and 1 as equal).
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def load_requests(directory):
    records = {}
    with (Path(directory) / "requests.jsonl").open() as source:
        for line in source:
            record = json.loads(line)
            if type(record["index"]) is not int or record["index"] < 0:
                raise ValueError("request index must be a non-negative integer")
            key = f"{record['phase']}/{record['workload']}/{record['index']}"
            if key in records:
                raise ValueError(f"duplicate request: {key}")
            raw_response = json.loads(bytes.fromhex(record["response_body_hex"]))
            if canonical(raw_response) != canonical(record["response"]):
                raise ValueError(
                    f"recorded JSON differs from original HTTP bytes: {key}"
                )
            records[key] = record
    return records


def compare(args):
    directories = (args.baseline, args.candidate)
    manifests = [
        json.loads((Path(directory) / "run.json").read_text())
        for directory in directories
    ]
    all_records = [load_requests(directory) for directory in directories]
    differences = []
    for field in (
        "concurrency",
        "warmup_per_workload",
        "socket_timeout_seconds",
        "requests_per_workload",
        "model",
        "workloads",
    ):
        if canonical(manifests[0][field]) != canonical(manifests[1][field]):
            differences.append(
                {
                    "error": "run configuration differs",
                    "field": field,
                    "baseline": manifests[0][field],
                    "candidate": manifests[1][field],
                }
            )
    for directory, manifest, records in zip(directories, manifests, all_records):
        expected = {
            f"{phase}/{workload}/{index}"
            for workload in manifest["workloads"]
            for phase, count in (
                ("warmup", manifest["warmup_per_workload"]),
                ("measured", manifest["requests_per_workload"]),
            )
            for index in range(count)
        }
        if records.keys() != expected:
            differences.append(
                {
                    "error": "request index set differs from planned workload",
                    "directory": directory,
                    "missing": sorted(expected - records.keys()),
                    "unexpected": sorted(records.keys() - expected),
                }
            )
        for key, record in records.items():
            if not record["success"] or record["status"] != 200 or record["errors"]:
                differences.append(
                    {"error": "failed request", "directory": directory, "request": key}
                )
            errors = validate_response(
                record["response"], record["workload"], manifest["model"]
            )
            if errors:
                differences.append(
                    {"error": "invalid response", "request": key, "details": errors}
                )
            planned = manifest["workloads"].get(record["workload"])
            if planned is None or canonical(
                [record["endpoint"], record["request"]]
            ) != canonical(planned):
                differences.append(
                    {
                        "error": "recorded request differs from planned workload",
                        "directory": directory,
                        "request": key,
                    }
                )
        if not json.loads((Path(directory) / "summary.json").read_text())["success"]:
            differences.append(
                {"error": "run failed, including warmup", "directory": directory}
            )
    before, after = all_records
    for key in sorted(before.keys() | after.keys()):
        left, right = before.get(key), after.get(key)
        if left is None or right is None:
            differences.append({"request": key, "error": "missing request"})
        elif not left["success"] or not right["success"]:
            differences.append(
                {
                    "request": key,
                    "error": "failed request",
                    "baseline_errors": left["errors"],
                    "candidate_errors": right["errors"],
                }
            )
        elif canonical(left["request"]) != canonical(right["request"]):
            differences.append({"request": key, "error": "request payload differs"})
        elif canonical(normalize(left["response"])) != canonical(
            normalize(right["response"])
        ):
            differences.append(
                {
                    "request": key,
                    "error": "response differs",
                    "baseline": normalize(left["response"]),
                    "candidate": normalize(right["response"]),
                }
            )
    if not any(r["phase"] == "measured" for r in before.values()) or not any(
        r["phase"] == "measured" for r in after.values()
    ):
        differences.append({"error": "no measured requests"})
    report = {
        "success": not differences,
        "ignored_fields": [
            "$.id",
            "$.created",
            "$.choices[*].message.tool_calls[*].id",
        ],
        "baseline_requests": len(before),
        "candidate_requests": len(after),
        "differences": differences,
    }
    with Path(args.output).open("x") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "differences"}
        )
    )
    return 0 if report["success"] else 1


def bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return number

    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Run fixed, non-streaming requests without retries"
    )
    run_parser.add_argument("--url", required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output", required=True, help="New result directory")
    run_parser.add_argument(
        "--workload", choices=["both", "completions", "chat-tools"], default="both"
    )
    run_parser.add_argument("--requests", type=bounded_int(1, 10000), default=8)
    run_parser.add_argument("--warmup", type=bounded_int(0, 100), default=2)
    run_parser.add_argument("--concurrency", type=bounded_int(1, 32), default=2)
    run_parser.add_argument("--timeout", type=bounded_int(1, 300), default=60)
    run_parser.add_argument(
        "--api-key-env", help="Name of environment variable; secret is not saved"
    )
    diff_parser = commands.add_parser(
        "compare", help="Strict full response equality, including warmup"
    )
    diff_parser.add_argument("--baseline", required=True)
    diff_parser.add_argument("--candidate", required=True)
    diff_parser.add_argument("--output", required=True, help="New JSON comparison file")
    args = parser.parse_args(argv)
    try:
        return run(args) if args.command == "run" else compare(args)
    except (OSError, ValueError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
