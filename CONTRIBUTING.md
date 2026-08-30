# Contributing to llm-meter

`llm-meter` is under active development. Contributions that advance inference
metrology — reproducible measurement, telemetry correlation, and explainable
benchmarks — are welcome.

## Focus

- Create **focused PRs**. One logical change per PR.
- **Include tests** for behavior changes. Run `ruff check .` and `pytest`
  before requesting review; both must pass.
- Keep the **core measurement model engine-neutral**. Engine-specific logic
  belongs in adapters, not in the core.

## Benchmark claims

- Benchmark claims require a **reproducible methodology**. State the workload,
  concurrency, token distributions, engine configuration, and environment.
- **Do not commit benchmark numbers** without environment and workload context.
- **Distinguish observed data from interpretation.** Raw measurements are
  facts; "GPU saturated" or "queueing bottleneck" are conclusions derived from
  those facts. Keep them separate.
- Benchmark provenance (model, engine, GPU, driver, CUDA, git SHA, etc.) is
  part of the measurement, not optional metadata.

## Scope

- Do not overclaim features that are not implemented. Use language such as
  "planned", "target", "under active development", or "V1 is being designed".
- `llm-meter` is inference metrology, not a load generator, model deployment
  platform, or GPU profiler. See the Non-goals section in the README.

## Development setup

```bash
uv sync
ruff check .
pytest
llm-meter --version
```
