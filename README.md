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

The metric vocabulary below defines the scope of `llm-meter`. Items marked **planned** are part of the target design and not yet implemented.

### Request / latency metrics

| Metric | Description | Status |
| --- | --- | --- |
| TTFT | Time To First Token — latency from request submission to first generated token | planned |
| TPOT | Time Per Output Token — average time between successive output tokens | planned |
| ITL | Inter-Token Latency — per-token decode timing distribution | planned |
| E2E latency | End-to-end request latency, submission to completion | planned |
| Queue wait | Time spent queued before prefill begins | planned |
| Request duration | Total wall-clock duration of a single request | planned |
| Prefill timing | Prefill-phase timing where exposed by the serving engine | planned |
| Decode timing | Decode-phase timing where exposed by the serving engine | planned |

### Throughput metrics

| Metric | Description | Status |
| --- | --- | --- |
| Output tokens/sec | Generated token throughput | planned |
| Input tokens/sec | Consumed prompt token throughput | planned |
| Total tokens/sec | Input + output token throughput | planned |
| Requests/sec | Completed request rate | planned |
| Completed requests | Count of successfully completed requests | planned |
| Failed requests | Count of failed requests | planned |

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

### System / GPU telemetry

All system and GPU telemetry is **planned** and not yet implemented.

| Telemetry | Description | Status |
| --- | --- | --- |
| GPU utilization | SM activity percentage | planned |
| GPU memory usage | Allocated / reserved GPU memory | planned |
| Power | GPU power draw | planned |
| Temperature | GPU temperature | planned |
| SM activity | Per-SM activity where available | planned |
| Memory bandwidth | Memory bandwidth related telemetry where available | planned |
| KV-cache utilization | KV-cache occupancy where exposed by the serving engine | planned |

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

Do not only retain averages or percentiles. The architecture should eventually permit retaining per-request and per-token measurements so reports can be regenerated and re-analyzed without re-running the benchmark.

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

`llm-meter` is under active development. V1 is being designed. No production benchmark capability is claimed yet.

`llm-meter` measures LLM inference performance in a way that can eventually answer not only *"how fast was it?"* but also *"under exactly what conditions, and why?"*