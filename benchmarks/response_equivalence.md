# Compare complete non-streaming HTTP responses

Use `benchmark_response_equivalence.py` to record a finite, deterministic cohort
against an already running server, then compare it with another configuration.
It supplements `vllm bench serve` with complete original response bodies and
strict field comparison. It does not start servers, change GPU state or retry
failed requests.

```bash
python benchmarks/benchmark_response_equivalence.py run \
  --url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B \
  --label eager --output eager-responses --requests 32 --warmup 4

# After restarting the server with the candidate configuration:
python benchmarks/benchmark_response_equivalence.py run \
  --url http://127.0.0.1:8000 --model Qwen/Qwen3-0.6B \
  --label graph --output graph-responses --requests 32 --warmup 4

python benchmarks/benchmark_response_equivalence.py compare \
  --baseline eager-responses --candidate graph-responses \
  --output comparison.json
```

Both output directories and the comparison file must be new. To compare a
single workload, pass `--workload completions` or `--workload chat-tools` to
both runs. Configure the same model snapshot, parser, template, generation
settings and batch invariance on each server. Tool-chat responses depend on
the model's tool support; this benchmark submits requests, never executes tools.
For an authenticated server, `--api-key-env VARIABLE_NAME` reads the key from
the environment without writing it to the evidence files.

The completion cohort requests exactly 128 output tokens and per-token
logprobs; the chat cohort exposes a weather tool with temperature zero and
thinking disabled. Both use non-streaming responses, seed zero and concurrency
two by default. Full requests, raw response bytes encoded as hex, parsed
responses, usage, status, errors and request latency are retained, including
warmups and malformed/error responses. Run summaries report failure counts,
wall time, throughput and nearest-rank p50/p95 latency.

Comparison checks the complete planned request set, matching configurations
and request bodies, HTTP success, response validity and original-body/parsed
JSON agreement. It compares every warmup and measured response. Only
`$.id`, `$.created` and `$.choices[*].message.tool_calls[*].id` are removed;
text, token/logprob fields, usage, finish reasons, tool arguments and other
fields remain strict. JSON types are preserved during comparison, so a boolean
cannot silently compare equal to an integer. Exit status is zero only on a
complete match; differences return one, malformed/incomplete evidence returns
two. A malformed response can fail while loading evidence before a comparison
report is created; its original bytes remain in the run directory.

A passing comparison proves equality for these recorded inputs and fields.
It does not prove CUDA Graph replay, general model quality or a performance
improvement. Capture a separate execution trace to verify replay. For timing,
use fresh processes in a predetermined paired order and report uncertainty
across independent process groups. Do not treat requests within a process as
independent runs. This non-streaming client does not measure TTFT, ITL or TPOT.
