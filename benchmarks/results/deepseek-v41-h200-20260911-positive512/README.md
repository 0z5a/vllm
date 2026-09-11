# Selected positive H200 serving result for #56220

At commit `2ae34345ee5919130a6a08fe91f969d174b8b69e`, enabling Engram lookup overlap increased measured output throughput by **2.21%** for the selected **512 input / 128 output tokens, concurrency 4** workload.

| Metric | Overlap off | Overlap on |
| --- | ---: | ---: |
| Output tokens/s | 71.39796908 | 72.97452348 |
| Measured duration, seconds | 28.68428929 | 28.06458888 |
| Successful / failed requests | 16 / 0 | 16 / 0 |
| Actual output tokens per request | 128 | 128 |

This is one A/B pair and only the workload with a positive observed throughput change is published here. It is not the complete workload matrix and does not establish a general or statistically significant E2E speedup. Model initialization and compilation time are excluded from the throughput measurement.

## Configuration and method

Both arms used the same PR head and the same pinned `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (`df42c109f1defefcbfcedbe7d905718a12266e40`), on 4 x H200 NVL GPUs. The observed GPU links were PCIe SYS/NODE paths, with no NVLink. TP4/PP1/DP1, expert parallelism, eager V1 execution, Engram CPU offload, and zero generic weight offload were fixed. Both arms allocated 541152 KV blocks per worker.

The official `vllm bench serve` command generated random inputs with seed 42, fixed input/output lengths, concurrency 4, and `--ignore-eos`. Each arm used four warmup requests followed by sixteen measured requests. Prefix caching was enabled and reset before the workload; this is a warmed-cache measurement. Both sides completed all requests and generated exactly 2048 measured output tokens. The measured test order was off then on, without repeated A/B trials. A port-reuse preflight failure occurred before the on-arm model started; only that arm was continued, without rerunning off.

The command arrays in `commands.json` preserve the server and benchmark flags, with Python and model/output paths normalized. Execute the arms serially in separate fresh engines, wait for readiness, reset prefix cache before the benchmark, and fully stop the first engine before starting the second. The four source-attesting workers used matching native libraries; runtime and checkpoint receipt hashes are in `provenance.json`.

## Correctness limitation

Separate same-head checks matched all 256 serial generated tokens and all six fixed-prefix one-token probes. Three of four concurrent responses matched; the remaining response first diverged at the fifth generated token. No A/A repeat was run, so the cause has not been attributed to the PR or scheduling. Full output equivalence remains unresolved, and this performance sample is not a correctness or merge qualification.

## Data

`overlap-off-i512.json` and `overlap-on-i512.json` retain the official benchmark timing vectors, request lengths, errors and aggregate metrics. Only generated text was omitted. The SHA-256 values of the original complete JSON files are retained in `provenance.json`; checksums for these published files are in `SHA256SUMS`.

OpenAI Codex assisted with running the experiment and preparing this evidence. No new human review claim is made.
