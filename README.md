# llm-meter

**Transparent, reproducible metrology for LLM inference.**

> Measure LLM inference. Explain where the time went.

`llm-meter` is an open-source inference metrology toolkit for measuring and explaining LLM serving performance.

It is designed to correlate request-level measurements such as TTFT, TPOT, inter-token latency, end-to-end latency, queueing time, and throughput with GPU and serving-runtime telemetry.

The goal is not merely to produce benchmark numbers. The goal is to make those numbers reproducible, comparable, and explainable.

`llm-meter` is **under active development**. V1 is being designed. This repository establishes the project contract, metric vocabulary, and architecture direction; it does not yet claim production benchmark capability.

---

## Why llm-meter?

LLM inference benchmarking results often report numbers such as:

- 700 tokens/s
- 350 ms p95 TTFT
- 22 ms TPOT
- 95% GPU utilization

but frequently omit enough context to answer:

- What exact workload produced the result?
- What concurrency was used?
- What were the prompt and output token distributions?
- Was the server already saturated?
- How much time was spent queueing?
- What happened to KV-cache utilization?
- What was GPU memory pressure?
- What engine configuration was used?
- What model, precision, driver, CUDA version, and GPU produced the result?
- Can another engineer reproduce the same measurement?
- Why did two benchmark runs differ?

`llm-meter` intends to make these conditions first-class benchmark artifacts. Benchmark provenance — model, engine, GPU, driver, CUDA, git SHA, workload, and run configuration — is part of the measurement, not optional metadata. Reproducible LLM benchmarks require that the *conditions* of a run be captured with the same rigor as the *results*.

---

## What llm-meter measures

`llm-meter` is about transparent inference metrology, so metrics are
distinguished by **where they can actually be observed**. Not all metrics are
available from a client benchmark alone; some require engine/runtime adapters
or GPU/system telemetry integrations.

All metrics below are **planned** — part of the target design and not yet
implemented.

### Client-observable measurements

Measurements available from an OpenAI-compatible client without modifying the
inference server:

| Metric | Description | Status |
| --- | --- | --- |
| Request start | Timestamp when the request was submitted | planned |
| Response/header timing | Timing captured in response headers where available | planned |
| First streamed response event | Timestamp of the first SSE event arrival | planned |
| Streamed chunk arrival timestamps | Per-chunk arrival timestamps during streaming | planned |
| Completion timestamp | Timestamp when the response completed | planned |
| HTTP/API errors | Error responses, status codes, and error timing | planned |
| E2E latency | End-to-end request latency, submission to completion | planned |
| Client-observed TTFT | Time To First Token — latency from request submission to first streamed response event | planned |

A generic client can safely measure **stream event timestamps** and
**inter-chunk latency** (time between successive SSE chunk arrivals).

#### ITL versus streaming chunk latency

Do **not** assume:

> OpenAI SSE chunk == generated token

A streamed response chunk may contain zero, one, or multiple token pieces
depending on engine, protocol, and detokenization behavior. Therefore a generic
client can safely measure stream-event timestamps and inter-chunk latency, but
must only label a measurement as true token-level **ITL** (Inter-Token Latency)
when a defensible token-to-timestamp relationship exists.

`llm-meter` preserves **raw stream-event timestamps** so that later adapters
with engine-specific knowledge can derive richer measurements (true per-token
ITL) without losing the original raw observations.

| Metric | Description | Status |
| --- | --- | --- |
| ITL | Inter-Token Latency — per-token decode timing, valid only when a defensible token-to-timestamp relationship exists | planned |
| Inter-chunk latency | Time between successive streamed chunk arrivals | planned |

### Token/count-derived measurements

These measurements require an explicit token-count source:

| Metric | Description | Status |
| --- | --- | --- |
| Input tokens | Prompt token count | planned |
| Output tokens | Generated token count | planned |
| Output tokens/sec | Generated token throughput | planned |
| Input tokens/sec | Consumed prompt token throughput | planned |
| Total tokens/sec | Input + output token throughput | planned |
| TPOT | Time Per Output Token, derived from token count + request timestamps | planned |

The eventual benchmark artifact **must record how token counts were obtained**.
Possible token-count sources include:

- server-reported usage (e.g. OpenAI `usage` field)
- engine-reported usage (e.g. vLLM metrics endpoint)
- locally tokenized approximation (client-side tokenizer)

**Never silently mix those sources.** A measurement derived from one source is
not directly comparable to a measurement from another without explicit context.

#### TPOT definition

For a completed generation with more than one output token, `llm-meter` uses
the following conventional client-derived definition:

```
TPOT = (E2E latency - TTFT) / (output_tokens - 1)
```

Important constraints:

- The **exact token-count source** must be recorded in the artifact.
- **Edge cases** such as `output_tokens <= 1` must be represented explicitly
  rather than producing misleading values (e.g. division by zero or a single
  token reporting zero TPOT).
- **Raw timestamps and counts remain the source of truth.** TPOT is a derived
  aggregate; the artifact should retain the underlying measurements so the
  aggregate can be regenerated or reinterpreted.

This formula is documented here as part of the V0 measurement contract. It is
not yet implemented in code.

### Server / runtime-observable measurements

These measurements **cannot be inferred reliably from a generic OpenAI-compatible
HTTP client**. They require an engine/runtime adapter or telemetry integration:

| Metric | Description | Status |
| --- | --- | --- |
| Queue wait | Time spent queued in the scheduler before prefill begins | planned |
| Prefill timing | Prefill-phase execution timing where exposed by the serving engine | planned |
| Decode timing | Decode-phase execution timing where exposed by the serving engine | planned |
| Scheduler state | Scheduler queue depth and batching state | planned |
| KV-cache occupancy | KV-cache memory occupancy where exposed by the serving engine | planned |
| KV-cache utilization | KV-cache utilization where exposed by the serving engine | planned |
| Batch composition | Active batch size and request composition | planned |
| Request duration | Total wall-clock duration of a single request within the engine | planned |

### GPU / system-observable measurements

These measurements require GPU/system telemetry sources such as
NVML, DCGM, CUPTI, or profiler integrations depending on the metric.

All GPU/system telemetry is **planned** and not yet implemented.

| Telemetry | Description | Status |
| --- | --- | --- |
| GPU utilization | Device busy/utilization metric exposed by the selected telemetry provider | planned |
| SM activity | Streaming-multiprocessor activity metric where supported | planned |
| GPU memory usage | Used / free / total GPU memory | planned |
| Memory activity/bandwidth | Memory activity or bandwidth metric where supported | planned |
| Power | GPU power draw | planned |
| Temperature | GPU temperature | planned |

"GPU utilization" and "SM activity" are **separate measurements**. GPU
utilization is a device-level busy/utilization metric exposed by the selected
telemetry provider; SM activity is a streaming-multiprocessor-level activity
metric where supported. They must not be conflated.

The benchmark artifact should eventually record the **telemetry source/provider**
(e.g. NVML, DCGM, CUPTI) alongside GPU measurements, so that the telemetry
provenance is explicit and reproducible.

### Workload dimensions

| Dimension | Description |
| --- | --- |
| Concurrency | Number of in-flight requests |
| Request rate | Target request submission rate |
| Prompt token length | Input token distribution |
| Requested output token length | Target generation length |
| Actual output token length | Observed generation length |
| Streaming / non-streaming | Whether responses are streamed |
| Warmup configuration | Warmup phase duration and shape |
| Run duration | Measured phase length |

### Environment provenance

The eventual benchmark artifact should capture:

- model
- model revision where available
- tokenizer
- inference engine
- engine version
- engine startup arguments
- GPU model
- GPU count
- GPU memory
- NVIDIA driver
- CUDA runtime/toolkit where available
- PyTorch version where relevant
- operating system
- container image
- git SHA
- llm-meter version

Benchmark provenance is part of the measurement, not optional metadata. A throughput number detached from the environment and workload that produced it is not a reproducible result.

---

## Design principles

### 1. Reproducibility over impressive numbers

A slower benchmark that can be reproduced is more useful than an unexplained throughput claim. Every reported number must carry enough context to be re-run by another engineer.

### 2. Preserve raw measurements

Do not only retain averages or percentiles. The architecture should eventually
permit retaining per-request, per-chunk, and per-token measurements so reports
can be regenerated and re-analyzed without re-running the benchmark. Raw
stream-event timestamps must be preserved so that later adapters can derive
richer measurements without losing the original observations.

### 3. Separate observation from interpretation

Raw measurements should remain independent from conclusions such as "GPU saturated" or "queueing bottleneck." Interpretation is derived from observation, never mixed into it.

### 4. Correlate client, server, and GPU behavior

The eventual architecture should allow a benchmark timeline to correlate:

```
request → queue → prefill → first token → decode → completion
```

with serving-runtime and GPU telemetry, so inference performance analysis can explain *where the time went*, not only *how long it took*.

### 5. Engine-neutral core

Do not tightly couple the core measurement model to any single inference engine. vLLM can be the first concrete integration, but adapters should allow other inference engines later.

Potential future adapters:

- vLLM — planned first integration
- SGLang — planned
- NVIDIA Triton — planned
- TensorRT-LLM — planned
- llama.cpp — planned
- other OpenAI-compatible inference endpoints — planned

None of these engines are currently supported. They are design targets only.

### 6. Machine-readable first

Every benchmark run should eventually produce a machine-readable artifact suitable for comparison, regression testing, visualization, CI, and historical analysis. Human-readable reports are derived from that data.

---

## V0 — Repository foundation

This initial commit marks **V0**, the bootstrap milestone. V0 contains:

- project definition
- package skeleton
- CLI skeleton
- metric vocabulary
- benchmark artifact schema direction
- testing/linting foundation
- contribution guidelines

V0 does **not** claim production benchmark capability. It establishes the contract and direction for the project.

## V1 direction

**V1 is being designed**, not yet implemented. The intended first useful version focuses on:

1. OpenAI-compatible inference endpoint
2. streaming requests
3. controlled concurrency
4. configurable prompt/input length
5. configurable output-token target
6. warmup phase
7. measured benchmark phase
8. TTFT
9. TPOT
10. ITL
11. E2E latency
12. throughput
13. error rate
14. p50 / p90 / p95 / p99
15. machine-readable JSON results
16. CSV export
17. environment/provenance capture
18. basic run-to-run comparison

The first engine-specific integration can target vLLM benchmarking while retaining an engine-neutral core.

---

## Current capability (experimental)

`llm-meter` can now:

- **perform a single streaming request** against an OpenAI-compatible endpoint,
  preserve raw observations, and serialize a versioned `BenchmarkRun` JSON
  artifact.
- **construct deterministic, tokenizer-aware workloads** — given a tokenizer and
  a target input-token count, generate a reproducible prompt from a built-in
  corpus.
- **record workload provenance** — the artifact records how the prompt was
  constructed, what tokenizer was used, the target vs. actual local token count,
  and a SHA-256 fingerprint of the prompt actually sent.
- **fingerprint prompts** — the exact prompt text actually sent is hashed
  (SHA-256) after final workload resolution, not before.
- **execute warmup and measured phases** — a benchmark session runs warmup
  requests followed by measured requests at a fixed concurrency, preserving
  every individual `BenchmarkRun`.
- **control concurrency** — a worker-pool limits the maximum number of
  simultaneously in-flight requests. Concurrency is not QPS, request rate, or
  batch size.
- **serialize a `BenchmarkSession` artifact** — the session artifact contains
  the execution plan, all per-request `BenchmarkRun` objects, and orchestration
  timing offsets.

This is an experimental multi-request capability. Percentile aggregation,
throughput summaries, and GPU telemetry are not yet implemented.

### Usage

```bash
export LLM_METER_API_KEY="your-api-key"   # optional, if the endpoint requires auth

# Mode A: manual prompt (optionally with a tokenizer for local token counting)
llm-meter run-one \
  --endpoint http://localhost:8000/v1 \
  --model some-model \
  --prompt "Explain dynamic batching briefly." \
  --max-output-tokens 64 \
  --output run.json

# Mode B: tokenizer-aware workload construction (reproducible prompt)
llm-meter run-one \
  --endpoint http://localhost:8000/v1 \
  --model some-model \
  --tokenizer Qwen/Qwen3-8B \
  --input-tokens 512 \
  --max-output-tokens 128 \
  --seed 42 \
  --output run.json
```

The command performs exactly one streaming request, saves a `BenchmarkRun`
artifact to the specified path, and prints a concise summary.

`--prompt` and `--input-tokens` are mutually exclusive. `--input-tokens`
requires `--tokenizer`. Manual `--prompt` may optionally specify `--tokenizer`
so that llm-meter can locally measure the prompt token count.

A manually supplied prompt has no input token target unless future APIs
explicitly introduce one. If a tokenizer is supplied, llm-meter records the
locally measured token count. That measured count is not a target-resolution
result. Manual mode may omit `--max-output-tokens`; in that case the HTTP
request simply omits `max_tokens`.

Hugging Face tokenizer loading (`--tokenizer Qwen/Qwen3-8B`) may require
network access or a populated local tokenizer cache. CI for llm-meter itself
remains network-free; tests use a deterministic `FakeTokenizer`.

### Workload specification

`llm-meter` supports two prompt input modes:

| Mode | Flags | Behavior |
| --- | --- | --- |
| Manual | `--prompt TEXT` (+ optional `--tokenizer`) | Use the supplied prompt text directly. If a tokenizer is provided, the local token count is measured and recorded. |
| Builtin | `--input-tokens N --tokenizer TOKENIZER` | Construct a deterministic, reproducible prompt from a built-in corpus, sized to approximately N tokens. |

For builtin workloads, the prompt is constructed deterministically from a
seeded shuffle of a neutral sentence corpus, then truncated to the target
token count. The final prompt is **re-encoded** after construction — the
actual local token count is authoritative, not the target.

Resolution status:

| Status | Meaning |
| --- | --- |
| `exact` | Local token count exactly equals the target (builtin only) |
| `nearest` | A valid prompt was produced but the local count differs from the target (builtin only) |
| `not_applicable` | Manual prompt — no input token target to resolve against |
| `unresolvable` | No valid prompt could be produced (reserved for future bounded construction failures) |

### `workload inspect`

Inspect a resolved workload specification without making any network request:

```bash
llm-meter workload inspect \
  --tokenizer Qwen/Qwen3-8B \
  --input-tokens 512 \
  --output-tokens 128 \
  --seed 42
```

`--tokenizer` is required. Add `--show-prompt` to print the full generated
prompt text. Hugging Face tokenizer loading may require network access or a
populated local tokenizer cache.

### BenchmarkRun artifact

The artifact is a JSON object with schema version `1`. It contains:

```json
{
  "schema_version": "1",
  "run_id": "uuid",
  "started_at": "2025-01-01T00:00:00+00:00",
  "configuration": {
    "endpoint": "http://localhost:8000/v1",
    "model": "some-model",
    "streaming": true,
    "max_output_tokens": 64
  },
  "request_start": {
    "offset_ns": 0,
    "wall_clock_utc": "2025-01-01T00:00:00+00:00"
  },
  "stream_events": [
    {
      "sequence": 0,
      "offset_ns": 12000000,
      "event_type": "metadata",
      "text_delta": null,
      "finish_reason": null,
      "usage": null
    },
    {
      "sequence": 1,
      "offset_ns": 45000000,
      "event_type": "content",
      "text_delta": "Hello",
      "finish_reason": null,
      "usage": null
    }
  ],
  "completion": {
    "offset_ns": 90000000,
    "wall_clock_utc": "2025-01-01T00:00:01+00:00"
  },
  "error": null,
  "usage": {
    "input_tokens": 5,
    "output_tokens": 10,
    "source": "server_reported"
  },
  "metrics": {
    "client_ttft_ns": 45000000,
    "e2e_latency_ns": 90000000,
    "inter_chunk_latencies_ns": [33000000, 15000000],
    "tpot_ns": 5000000,
    "tpot_status": "ok"
  },
  "provenance": {
    "llm_meter_version": "0.1.0.dev0"
  },
  "workload": {
    "source": "builtin",
    "seed": 42,
    "input_tokens_target": 512,
    "output_tokens_target": 128,
    "input_tokens_actual_local": 510,
    "resolution_status": "nearest",
    "prompt_sha256": "ab12cd34...",
    "prompt_chars": 2104,
    "tokenizer_provider": "huggingface",
    "tokenizer_id": "Qwen/Qwen3-8B",
    "tokenizer_revision": null
  }
}
```

All timestamps are **integer nanoseconds** relative to an explicit run origin.
Absent data is represented as `null` rather than fabricated. No API keys or
secrets appear in the artifact. Prompt text is **not** persisted in the
artifact — only its SHA-256 fingerprint and character count.

### `run-batch` — multi-request benchmark

Execute warmup requests followed by measured requests at a fixed concurrency:

```bash
llm-meter run-batch \
  --endpoint http://localhost:8000/v1 \
  --model some-model \
  --tokenizer Qwen/Qwen3-8B \
  --input-tokens 512 \
  --max-output-tokens 128 \
  --warmup-requests 4 \
  --requests 20 \
  --concurrency 4 \
  --seed 42 \
  --output session.json
```

Manual prompt mode:

```bash
llm-meter run-batch \
  --endpoint http://localhost:8000/v1 \
  --model some-model \
  --prompt "Explain dynamic batching." \
  --max-output-tokens 128 \
  --warmup-requests 2 \
  --requests 10 \
  --concurrency 2 \
  --output session.json
```

#### Concurrency is not QPS, request rate, or batch size

**Concurrency** is the maximum number of requests simultaneously in flight.
It does not mean:

- **QPS / request rate** — how many requests per second to submit
- **batch size** — the serving engine's internal batch composition
- **arrival distribution** — a stochastic load model

llm-meter does not implement request-rate pacing. All eligible requests are
submitted as fast as the concurrency limit allows. A future PR may introduce
rate-controlled arrival.

#### Warmup phase

Warmup requests are **real requests**. They may initialize connection pools,
serving-runtime state, kernels, caches, and memory allocations.

Warmup request metrics are **never** treated as measured benchmark samples.
Every request carries an explicit `phase` field (`warmup` or `measured`), and
the session artifact provides separated `warmup_runs` / `measured_runs`
collections.

All warmup requests complete before any measured request begins. There is no
overlap between phases.

#### Per-request seed strategy (builtin)

For builtin generated workloads, each request gets a deterministic per-request
seed:

```
request_seed = base_seed + global_request_ordinal
```

where the global ordinal spans both phases:

```
warmup #0        → seed + 0
warmup #1        → seed + 1
...
measured #0      → seed + warmup_requests
measured #1      → seed + warmup_requests + 1
```

This means measured prompts do not intentionally reuse warmup prompt seeds.
The final `prompt_sha256` is authoritative — the seed strategy does not
guarantee prompt uniqueness, but makes it deterministic.

The session records the seed strategy as `base_plus_global_ordinal` in
`configuration.seed_strategy`.

#### Manual prompt caveat

For manual prompt workloads, the same prompt text is reused across all
requests (warmup and measured). A repeated manual prompt may interact with
server-side prefix caching, which can affect observed latency. This is
documented behavior, not a bug — the artifact records the prompt fingerprint so
the condition is reproducible.

### BenchmarkSession artifact

The session artifact is a JSON object with schema version `1`. It contains:

```json
{
  "schema_version": "1",
  "session_id": "uuid",
  "started_at": "2025-01-01T00:00:00+00:00",
  "completed_at": "2025-01-01T00:01:00+00:00",
  "status": "completed",
  "configuration": {
    "endpoint": "http://localhost:8000/v1",
    "model": "some-model",
    "warmup_requests": 4,
    "measured_requests": 20,
    "concurrency": 4,
    "seed": 42,
    "seed_strategy": "base_plus_global_ordinal",
    "max_connections": 4,
    "max_keepalive_connections": 4,
    "prompt_source": "builtin",
    "input_tokens_target": 512,
    "output_tokens_target": 128,
    "tokenizer_id": "Qwen/Qwen3-8B",
    "max_output_tokens": 128
  },
  "requests": [
    {
      "phase": "warmup",
      "ordinal": 0,
      "session_start_offset_ns": 1000,
      "session_finish_offset_ns": 50000000,
      "run": { "...": "BenchmarkRun" }
    }
  ],
  "provenance": {
    "llm_meter_version": "0.1.0.dev0"
  }
}
```

Session status semantics:

| Status | Meaning |
| --- | --- |
| `completed` | The runner executed all planned requests. Individual request failures are recorded per-run, not at session level. |
| `failed` | A runner-level/internal failure prevented completion of the plan. |

No API keys, secrets, or prompt text appear in the session artifact. Each
nested `BenchmarkRun` preserves its own `WorkloadProvenance` and server `Usage`
independently.

---

## Measurement semantics

These definitions are part of the public contract.

### Monotonic vs wall clock

- **Monotonic clock** (`time.perf_counter_ns()`): used for all elapsed-time
  and latency measurements. Stored as integer nanoseconds (`offset_ns`,
  `duration_ns`) relative to an explicit request/run origin. Monotonic clocks
  are not affected by system time adjustments.
- **Wall clock** (UTC ISO 8601): stored for human correlation and provenance
  only. Never used for latency calculation. Wall-clock timestamps may drift or
  jump; monotonic timestamps do not.

### client-observed TTFT

Time from the measured client request start until the first **content-bearing**
streamed generation event. Metadata-only events (e.g. role assignment) are not
counted. If no content-bearing event arrives, `client_ttft_ns` is `null`.

This is a **client-observed** metric. It is not the same as server-side prefill
latency, which requires engine/runtime instrumentation.

### inter-chunk latency

Time between successive streamed chunk arrivals. A streamed SSE chunk may
contain zero, one, or multiple token pieces depending on engine, protocol, and
detokenization behavior. Therefore `inter_chunk_latency` is **not** the same as
token-level ITL (Inter-Token Latency).

True ITL can only be derived when an adapter provides a defensible
token-to-timestamp mapping. `llm-meter` preserves raw stream-event timestamps
so that later adapters can derive true ITL without losing raw observations.

### E2E latency

```
E2E = completion_offset_ns - request_start_offset_ns
```

Measured entirely from the monotonic clock. If the request did not complete
(error or unexpected end), `e2e_latency_ns` is `null`.

### TPOT

For a completed generation with more than one output token:

```
TPOT = (E2E - client_ttft) / (output_tokens - 1)
```

Constraints:

- The **exact token-count source** must be recorded in the artifact.
- `output_tokens <= 1` → TPOT is `null` with status `insufficient_tokens`.
- No token count available → TPOT is `null` with status `no_token_count`.
- Chunk count is **never** substituted for token count.
- Raw timestamps and counts remain the source of truth; TPOT is a derived
  aggregate.

### token-count source

The artifact records how token counts were obtained:

| Source | Description |
| --- | --- |
| `server_reported` | From the OpenAI-compatible `usage` field in the response |
| `engine_reported` | From an engine-specific metrics endpoint (planned) |
| `locally_tokenized` | From a client-side tokenizer approximation (planned) |
| `unknown` | No token count available |

Never silently mix sources. A measurement from one source is not directly
comparable to a measurement from another without explicit context.

### local prompt tokens vs. server prompt tokens

`llm-meter` measures a **local prompt token count**
(`workload.input_tokens_actual_local`) using the configured tokenizer with
`add_special_tokens=False`. This counts the tokens of the **user prompt text
payload** — the raw string sent in the `messages` array.

This is **not** the same as `usage.input_tokens` (server-reported
`prompt_tokens`). The server may apply additional transformations before
prefill:

- **chat templates** that wrap the user message with role markers
- **BOS / EOS** or other special tokens
- **system prompts** prepended by the serving engine
- **tokenization differences** between the local tokenizer and the server's
  tokenizer

Therefore `workload.input_tokens_actual_local` and `usage.input_tokens` are
**independent measurements** that may legitimately differ. Both are preserved
in the artifact. Neither overwrites the other.

---

## Non-goals

For early versions, `llm-meter` is **not**:

- an LLM evaluation/quality framework
- a prompt evaluation system
- an inference router
- an API gateway
- a model deployment platform
- an autoscaler
- a full GPU profiler
- a replacement for Nsight Systems
- a replacement for PyTorch Profiler
- a replacement for Prometheus/Grafana
- a distributed fleet scheduler

It may consume or correlate information from such systems later. This scope boundary is important: `llm-meter` is inference metrology, not a general-purpose MLOps platform.

---

## Future CLI experience

The CLI interface below is **intended interface / roadmap** and is not yet implemented. It is shown to communicate design direction only.

```bash
llm-meter run \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B \
  --concurrency 1,4,8,16,32 \
  --input-tokens 512 \
  --output-tokens 128
```

Potential conceptual output:

```text
C     TTFT p50   TTFT p95   TPOT p95   req/s   output tok/s
1        ...        ...         ...      ...        ...
4        ...        ...         ...      ...        ...
8        ...        ...         ...      ...        ...
```

Future concepts also include:

```bash
llm-meter compare run-a.json run-b.json
```

and:

```bash
llm-meter inspect run.json
```

These commands do not exist yet. They are roadmap targets for V1 and beyond.

---

## Benchmark artifact philosophy

A benchmark should eventually be represented as an immutable run artifact. Conceptually:

```text
BenchmarkRun
├── metadata
├── environment
├── model
├── engine
├── workload
├── requests
├── stream_events
├── token_timings
├── gpu_samples
├── engine_samples
├── aggregates
└── provenance
```

The artifact is the source of truth. Human-readable reports, comparisons, and regressions are all derived from it. This keeps observation separate from interpretation and enables reproducible LLM benchmarks.

The schema is not finalized in this first commit. This section communicates the design direction only.

---

## Tooling

`llm-meter` uses Python 3.11+ with `uv` for environment and package management, `pytest` for tests, and `ruff` for linting.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

## Status

`llm-meter` is under active development. The experimental commands `run-one`
and `run-batch` can perform single-request and multi-request benchmarks,
producing `BenchmarkRun` and `BenchmarkSession` JSON artifacts with
deterministic, tokenizer-aware workload specification and prompt fingerprinting.
The `workload inspect` command resolves a workload specification without making
any network request.

Implemented:

- single-request observation
- workload specification
- warmup phase
- measured phase
- controlled concurrency
- per-request `BenchmarkRun` preservation
- `BenchmarkSession` artifact

Still planned:

- percentile aggregation (p50, p90, p95, p99)
- aggregate throughput
- error-rate summary
- environment/GPU provenance
- CSV export
- run-to-run comparison
- GPU telemetry
- engine-specific adapters

V1 is being designed.

`llm-meter` measures LLM inference performance in a way that can eventually answer not only *"how fast was it?"* but also *"under exactly what conditions, and why?"*
