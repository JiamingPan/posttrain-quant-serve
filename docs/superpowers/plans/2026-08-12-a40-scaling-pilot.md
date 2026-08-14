# A40-Constrained FSDP2 Scaling Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly labelled pilot mode that validates the scaling machinery on one and two A40s and records one honest two-A40 Qwen3-8B measurement without weakening the production 1/2/4/8 benchmark.

**Architecture:** Extend the existing `bench.scaling` configuration, worker, controller, and append-only records rather than creating a second benchmark. Keep `validate_scaling_records()` strict for the production four-point sweep; add a separate pilot validator and controller request checks for requested subsets. Let the existing controller launch isolated torchrun workers sequentially on the first N visible GPUs, and teach the existing Slurm launcher to permit a non-eight-GPU controller only when `--pilot` is explicit.

**Tech Stack:** Python 3.11, PyTorch FSDP2/DTensor, `torchrun`, `torch.profiler`, pytest, Bash, Slurm, existing repository run tracking.

## Global Constraints

- Default/full mode still requires exactly world sizes `1,2,4,8` on eight homogeneous visible GPUs.
- Pilot mode is opt-in through `--pilot`; it never changes full-mode validation.
- Use fixed global batch size 8, local microbatch size 1, 3 warmup optimizer steps, 10 unprofiled measured optimizer steps, and 3 separately profiled optimizer steps.
- The 0.5B harness experiment uses sequence length 512; the Qwen3-8B feasibility experiment uses sequence length 2048. Never compare their performance metrics.
- Treat the 0.5B 1-to-2 result as a communication-heavy small-batch strong-scaling lower bound, not a production scaling claim.
- Qwen3-8B pilot records require a clean Git worktree, a committed passing SFT gate, and an explicit immutable model revision.
- Qwen3-8B world-size-2 `scaling_efficiency` is `null` because there is no valid world-size-1 A40 baseline.
- Never present an analytical prediction as a measured value.
- Do not submit Slurm work without a separate exact preview and `APPROVE RUN`.
- Maximum approved design envelope for a later submission is two A40 GPUs for one hour, or two GPU-hours.

---

### Task 1: Define the Pilot CLI and Record Contract

**Files:**
- Modify: `tests/test_cli_contracts.py`
- Modify: `tests/test_scaling_metrics.py`
- Modify: `bench/scaling.py`

**Interfaces:**
- Consumes: `ScalingConfig`, `parse_scaling_args()`, `SCALING_FIELDS`, `RANK_FIELDS`, and the existing record dictionaries in `run_worker()`.
- Produces: `ScalingConfig.pilot: bool`, `ScalingConfig.benchmark_mode: str`, `benchmark_mode` in scaling and rank records, and `validate_pilot_scaling_records(records, *, expected_world_sizes)`.

- [ ] **Step 1: Write failing CLI and record-mode tests**

Add to `tests/test_cli_contracts.py`:

```python
def test_scaling_pilot_is_explicit_and_keeps_measurement_defaults() -> None:
    full = parse_scaling_args(["--output_dir", "/tmp/full"])
    pilot = parse_scaling_args(
        [
            "--pilot",
            "--world_sizes",
            "1,2",
            "--output_dir",
            "/tmp/pilot",
        ]
    )

    assert full.pilot is False
    assert full.benchmark_mode == "full"
    assert pilot.pilot is True
    assert pilot.benchmark_mode == "pilot"
    assert pilot.world_sizes == (1, 2)
    assert (pilot.warmup_steps, pilot.measure_steps, pilot.profile_steps) == (3, 10, 3)
```

Update `_valid_record()` in `tests/test_scaling_metrics.py` so it includes:

```python
"benchmark_mode": "full",
"measure_steps": 10,
"mfu": 0.25,
"scaling_efficiency": 1.0,
"step_time_mean_seconds": 1.0,
"step_time_std_seconds": 0.05,
```

and every rank record includes:

```python
"step_seconds": [1.0] * 10,
```

- [ ] **Step 2: Write failing pilot-validation tests**

Import `validate_pilot_scaling_records` and add:

```python
def test_pilot_validation_accepts_a_two_point_lower_bound() -> None:
    records = [_valid_record(world_size) for world_size in (1, 2)]
    for record in records:
        record["benchmark_mode"] = "pilot"
    records[0]["scaling_efficiency"] = 1.0
    records[1]["scaling_efficiency"] = 0.75

    validate_pilot_scaling_records(records, expected_world_sizes=(1, 2))


def test_isolated_pilot_point_requires_null_efficiency() -> None:
    record = _valid_record(2)
    record["benchmark_mode"] = "pilot"
    record["scaling_efficiency"] = None

    validate_pilot_scaling_records([record], expected_world_sizes=(2,))

    record["scaling_efficiency"] = 1.0
    with pytest.raises(ValueError, match="world-size-1 baseline"):
        validate_pilot_scaling_records([record], expected_world_sizes=(2,))
```

Add parameterized rejection tests for `float("nan")` and `float("inf")` in
`tokens_per_sec`, `mfu`, `communication_active_fraction`,
`communication_exposed_fraction`, and `step_time_mean_seconds`. Add one test
that supplies nine rank `step_seconds` while `measure_steps == 10` and expects
a failure mentioning measured step durations.

- [ ] **Step 3: Run the focused tests to verify RED**

Run:

```bash
pytest -q tests/test_cli_contracts.py tests/test_scaling_metrics.py
```

Expected: failures because `--pilot`, `benchmark_mode`, and
`validate_pilot_scaling_records()` do not exist.

- [ ] **Step 4: Implement the minimal pilot data contract**

In `bench/scaling.py`:

1. Add `"benchmark_mode"` to both `SCALING_FIELDS` and `RANK_FIELDS`.
2. Add `pilot: bool = False` to `ScalingConfig`.
3. Add this property without serializing a second source of truth:

```python
@property
def benchmark_mode(self) -> str:
    return "pilot" if self.pilot else "full"
```

4. Add `parser.add_argument("--pilot", action="store_true")`.
5. Add `"benchmark_mode": config.benchmark_mode` to the rank and scaling
   record dictionaries in `run_worker()`.

Refactor common record validation into:

```python
def _validate_scaling_record_set(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_world_sizes: tuple[int, ...],
    benchmark_mode: str,
) -> None:
    if (
        not expected_world_sizes
        or tuple(sorted(set(expected_world_sizes))) != expected_world_sizes
        or any(size not in SUPPORTED_WORLD_SIZES for size in expected_world_sizes)
    ):
        raise ValueError("expected world sizes must be a strictly increasing supported subset")
    by_world_size = {int(record["world_size"]): record for record in records}
    if tuple(sorted(by_world_size)) != expected_world_sizes or len(records) != len(expected_world_sizes):
        raise ValueError("scaling records do not match the requested world sizes")
    if {str(record["benchmark_mode"]) for record in records} != {benchmark_mode}:
        raise ValueError("scaling record benchmark mode does not match its controller")
    if len({str(record["gpu_name"]) for record in records}) != 1:
        raise ValueError("scaling records require homogeneous GPU hardware")
    if len(records) > 1 and len(
        {str(record["comparison_config_digest"]) for record in records}
    ) != 1:
        raise ValueError("scaling records have mismatched comparison configurations")
    if {int(record["global_batch_size"]) for record in records} != {8}:
        raise ValueError("scaling records must use fixed global batch size eight")
    for world_size, record in by_world_size.items():
        for name in ("tokens_per_sec", "mfu", "step_time_mean_seconds"):
            value = float(record[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        step_std = float(record["step_time_std_seconds"])
        if not math.isfinite(step_std) or step_std < 0:
            raise ValueError("step_time_std_seconds must be finite and non-negative")
        for name in ("communication_active_fraction", "communication_exposed_fraction"):
            value = float(record[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        rank_memory = list(record["rank_memory"])
        if {int(rank["rank"]) for rank in rank_memory} != set(range(world_size)):
            raise ValueError(f"world size {world_size} is missing complete rank memory records")
        measure_steps = int(record["measure_steps"])
        for rank in rank_memory:
            allocated = int(rank["peak_allocated_bytes"])
            reserved = int(rank["peak_reserved_bytes"])
            durations = [float(value) for value in rank["step_seconds"]]
            if allocated <= 0 or reserved < allocated:
                raise ValueError("rank memory counters are invalid")
            if len(durations) != measure_steps or any(
                not math.isfinite(value) or value <= 0 for value in durations
            ):
                raise ValueError("rank record has invalid measured step durations")
```

The shared validator must require exact requested world-size coverage,
homogeneous hardware, matching comparison digests for multi-point sets, global
batch 8, finite positive throughput/MFU/mean step time, finite non-negative
step-time standard deviation, finite communication fractions in `[0, 1]`,
complete rank IDs, positive allocated memory, reserved memory covering
allocated memory, and exactly `measure_steps` finite positive rank durations.

Keep the production entry point strict:

```python
def validate_scaling_records(records: Sequence[Mapping[str, Any]]) -> None:
    _validate_scaling_record_set(
        records,
        expected_world_sizes=SUPPORTED_WORLD_SIZES,
        benchmark_mode="full",
    )
    for record in records:
        efficiency = float(record["scaling_efficiency"])
        if not math.isfinite(efficiency) or efficiency <= 0:
            raise ValueError("full scaling efficiency must be finite and positive")
```

Add the separate pilot entry point:

```python
def validate_pilot_scaling_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_world_sizes: tuple[int, ...],
) -> None:
    _validate_scaling_record_set(
        records,
        expected_world_sizes=expected_world_sizes,
        benchmark_mode="pilot",
    )
    has_baseline = 1 in expected_world_sizes
    for record in records:
        efficiency = record["scaling_efficiency"]
        if has_baseline:
            if efficiency is None or not math.isfinite(float(efficiency)) or float(efficiency) <= 0:
                raise ValueError("pilot scaling efficiency requires a positive world-size-1 baseline")
        elif efficiency is not None:
            raise ValueError("scaling efficiency requires a world-size-1 baseline")
```

Do not allow an empty, duplicated, unsupported, or differently ordered
`expected_world_sizes` tuple.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```bash
pytest -q tests/test_cli_contracts.py tests/test_scaling_metrics.py
```

Expected: all tests pass.

- [ ] **Step 6: Preview and create the first implementation commit**

Before committing, show the exact staged diff and the command:

```bash
git add bench/scaling.py tests/test_cli_contracts.py tests/test_scaling_metrics.py docs/superpowers/plans/2026-08-12-a40-scaling-pilot.md
git commit -m "feat: add pilot scaling record contract"
```

Proceed only after a new `APPROVE RUN`.

---

### Task 2: Generalize the Existing Controller Without Weakening Full Mode

**Files:**
- Create: `tests/test_scaling_controller.py`
- Modify: `bench/scaling.py`

**Interfaces:**
- Consumes: `ScalingConfig`, `_worker_cli_arguments()`, `build_worker_command()`, `_visible_gpu_ids()`, `_require_committed_correctness_gates()`, and both record validators.
- Produces: `required_correctness_gate_names(config) -> tuple[str, ...]`, `required_controller_gpu_count(config) -> int`, `apply_scaling_efficiencies(records) -> None`, and pilot-aware `run_sweep_controller()` / `validate_scaling_directory()`.

- [ ] **Step 1: Write failing pure controller-policy tests**

Create `tests/test_scaling_controller.py` with:

```python
from pathlib import Path

import pytest

from bench.scaling import (
    ScalingConfig,
    apply_scaling_efficiencies,
    build_worker_command,
    required_controller_gpu_count,
    required_correctness_gate_names,
)


def _config(tmp_path: Path, **overrides: object) -> ScalingConfig:
    values: dict[str, object] = {
        "output_dir": str(tmp_path),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    values.update(overrides)
    return ScalingConfig(**values)


def test_full_controller_still_requires_the_complete_eight_gpu_sweep(tmp_path: Path) -> None:
    assert required_controller_gpu_count(_config(tmp_path)) == 8
    with pytest.raises(ValueError, match="1,2,4,8"):
        required_controller_gpu_count(_config(tmp_path, world_sizes=(1, 2)))


def test_pilot_controller_requires_only_its_largest_world_size(tmp_path: Path) -> None:
    assert required_controller_gpu_count(
        _config(tmp_path, pilot=True, world_sizes=(1, 2))
    ) == 2
    assert required_controller_gpu_count(
        _config(tmp_path, pilot=True, world_sizes=(2,))
    ) == 2


def test_qwen3_pilot_requires_only_the_committed_sft_gate(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        pilot=True,
        world_sizes=(2,),
        model="Qwen/Qwen3-8B",
        revision="0123456789abcdef0123456789abcdef01234567",
    )

    assert required_correctness_gate_names(config) == ("sft_gate.json",)


def test_qwen3_full_sweep_retains_both_gate_requirements(tmp_path: Path) -> None:
    config = _config(tmp_path, model="Qwen/Qwen3-8B")

    assert required_correctness_gate_names(config) == (
        "sft_gate.json",
        "grpo_gate.json",
    )


def test_worker_command_propagates_pilot_mode(tmp_path: Path) -> None:
    command = build_worker_command(
        _config(tmp_path, pilot=True, world_sizes=(1, 2)),
        world_size=1,
        output_dir=tmp_path / "w1",
    )

    assert "--pilot" in command


def test_efficiency_is_null_without_world_size_one() -> None:
    records = [{"world_size": 2, "tokens_per_sec": 200.0, "scaling_efficiency": None}]

    apply_scaling_efficiencies(records)

    assert records[0]["scaling_efficiency"] is None


def test_efficiency_uses_world_size_one_when_present() -> None:
    records = [
        {"world_size": 1, "tokens_per_sec": 100.0, "scaling_efficiency": None},
        {"world_size": 2, "tokens_per_sec": 150.0, "scaling_efficiency": None},
    ]

    apply_scaling_efficiencies(records)

    assert records[0]["scaling_efficiency"] == pytest.approx(1.0)
    assert records[1]["scaling_efficiency"] == pytest.approx(0.75)
```

Add a test that a Qwen3 pilot without `revision` fails before worker launch,
and a test that duplicated or descending pilot world-size tuples fail.

- [ ] **Step 2: Run controller tests to verify RED**

Run:

```bash
pytest -q tests/test_scaling_controller.py
```

Expected: import failures for the new controller-policy functions.

- [ ] **Step 3: Implement controller policy helpers**

In `bench/scaling.py`, implement:

```python
def required_controller_gpu_count(config: ScalingConfig) -> int:
    if not config.pilot:
        if tuple(config.world_sizes) != SUPPORTED_WORLD_SIZES:
            raise ValueError("the full scaling controller requires world sizes 1,2,4,8")
        return 8
    if (
        len(set(config.world_sizes)) != len(config.world_sizes)
        or tuple(sorted(config.world_sizes)) != tuple(config.world_sizes)
    ):
        raise ValueError("pilot world sizes must be strictly increasing")
    return max(config.world_sizes)


def required_correctness_gate_names(config: ScalingConfig) -> tuple[str, ...]:
    if config.model != "Qwen/Qwen3-8B":
        return ()
    if config.pilot:
        return ("sft_gate.json",)
    return ("sft_gate.json", "grpo_gate.json")


def apply_scaling_efficiencies(records: list[dict[str, Any]]) -> None:
    baseline_record = next(
        (record for record in records if int(record["world_size"]) == 1),
        None,
    )
    if baseline_record is None:
        for record in records:
            record["scaling_efficiency"] = None
        return
    baseline = float(baseline_record["tokens_per_sec"])
    for record in records:
        record["scaling_efficiency"] = scaling_efficiency(
            throughput=float(record["tokens_per_sec"]),
            baseline_throughput=baseline,
            world_size=int(record["world_size"]),
        )
```

Refactor `_require_committed_correctness_gates()` to accept the exact gate-name
tuple returned by `required_correctness_gate_names()`. Require an explicit
revision for a Qwen3 pilot before capturing controller identity or spawning a
worker.

- [ ] **Step 4: Make worker command generation and allocation checks pilot-aware**

Append `--pilot` in `_worker_cli_arguments()` only when `config.pilot` is true.

In `run_sweep_controller()`:

1. call `required_controller_gpu_count(config)`;
2. require the exact gate set for Qwen3;
3. preserve the existing clean-worktree rule;
4. require both `_visible_gpu_ids()` and `torch.cuda.device_count()` to equal
   the required count;
5. require one identical GPU name across that count;
6. launch requested world sizes sequentially on the first N device IDs;
7. call `apply_scaling_efficiencies(records)`;
8. call `validate_pilot_scaling_records(records, expected_world_sizes=config.world_sizes)`
   in pilot mode, otherwise call `validate_scaling_records()`; and
9. write `benchmark_mode` into controller configuration and every consolidated
   record.

Update `validate_scaling_directory()` to read `run_config.json`. If it contains
`pilot: true`, validate against the recorded `world_sizes`; otherwise retain the
four-point validator. Reject a missing or inconsistent run config rather than
inferring the requested pilot subset from whatever files happen to exist.

- [ ] **Step 5: Run focused controller and metric tests to verify GREEN**

Run:

```bash
pytest -q tests/test_scaling_controller.py tests/test_scaling_metrics.py tests/test_cli_contracts.py
```

Expected: all tests pass.

- [ ] **Step 6: Preview and create the controller commit**

Show the exact diff and request approval for:

```bash
git add bench/scaling.py tests/test_scaling_controller.py tests/test_scaling_metrics.py tests/test_cli_contracts.py
git commit -m "feat: orchestrate partial A40 scaling pilots"
```

Proceed only after a new `APPROVE RUN`.

---

### Task 3: Allow an Explicit Pilot Through the Existing Slurm Launcher

**Files:**
- Modify: `tests/test_launch_slurm.py`
- Modify: `scripts/launch_slurm.sh`

**Interfaces:**
- Consumes: `scripts/launch_slurm.sh --stage scaling --gpus COUNT -- STAGE_ARGS`.
- Produces: full mode still requires `--gpus 8`; `--pilot` permits a scaling controller with `--gpus 1`, `2`, or `4` as well as `8`, with the Python controller performing exact subset/allocation validation.

- [ ] **Step 1: Write failing launcher tests**

Add to `tests/test_launch_slurm.py`:

```python
def test_scaling_pilot_controller_accepts_a_two_gpu_allocation() -> None:
    result = run_launcher(
        "--stage",
        "scaling",
        "--gpus",
        "2",
        "--dry-run",
        "--",
        "--pilot",
        "--world_sizes",
        "1,2",
        "--output_dir",
        "/tmp/a40-pilot",
    )

    assert result.returncode == 0, result.stderr
    assert "python -m bench.scaling" in result.stdout
    assert "stage_arg[0]=--pilot" in result.stdout
    assert "stage_arg[2]=1,2" in result.stdout


def test_scaling_controller_still_rejects_two_gpus_without_pilot() -> None:
    result = run_launcher(
        "--stage",
        "scaling",
        "--gpus",
        "2",
        "--dry-run",
        "--",
        "--world_sizes",
        "1,2",
        "--output_dir",
        "/tmp/not-a-pilot",
    )

    assert result.returncode != 0
    assert "requires --gpus 8 unless --pilot is explicit" in result.stderr
```

- [ ] **Step 2: Run launcher tests to verify RED**

Run:

```bash
pytest -q tests/test_launch_slurm.py
```

Expected: the two-GPU pilot is rejected by the current unconditional
eight-GPU controller check.

- [ ] **Step 3: Implement the minimal launcher exception**

Move `has_stage_argument()` above the stage-selection `case` so it can be used
while validating `scaling`. Replace the unconditional check with:

```bash
if ! has_stage_argument "--pilot"; then
  [[ "$GPUS" == "8" ]] || fail \
    "the complete scaling controller requires --gpus 8 unless --pilot is explicit"
fi
```

Do not infer pilot mode from a reduced GPU count or a reduced `--world_sizes`
argument. The literal `--pilot` flag is mandatory. Keep the existing visible
GPU count and single-node checks.

- [ ] **Step 4: Run launcher and CLI tests to verify GREEN**

Run:

```bash
pytest -q tests/test_launch_slurm.py tests/test_cli_contracts.py
bash scripts/launch_slurm.sh --stage scaling --gpus 2 --dry-run -- --pilot --world_sizes 1,2 --output_dir /tmp/a40-pilot
```

Expected: tests pass and the dry run prints a Python controller command with
`--pilot --world_sizes 1,2`; it does not print torchrun at the controller level.

- [ ] **Step 5: Preview and create the launcher commit**

Show the exact diff and request approval for:

```bash
git add scripts/launch_slurm.sh tests/test_launch_slurm.py
git commit -m "feat: launch scaling pilots on partial allocations"
```

Proceed only after a new `APPROVE RUN`.

---

### Task 4: Verify the Pilot Locally and Publish the Implementation Branch

**Files:**
- Verify: `bench/scaling.py`
- Verify: `scripts/launch_slurm.sh`
- Verify: `tests/test_scaling_controller.py`
- Verify: `tests/test_scaling_metrics.py`
- Verify: `tests/test_cli_contracts.py`
- Verify: `tests/test_launch_slurm.py`

**Interfaces:**
- Consumes: all pilot implementation commits.
- Produces: a clean, tested branch ready for a separately approved push and Great Lakes pull.

- [ ] **Step 1: Run static and focused verification**

Run:

```bash
python -m compileall -q bench train scripts
bash -n scripts/launch_slurm.sh
pytest -q tests/test_scaling_controller.py tests/test_scaling_metrics.py tests/test_cli_contracts.py tests/test_launch_slurm.py
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the complete local test suite**

Run:

```bash
pytest -q
```

Expected: all CPU-capable tests pass; CUDA/distributed tests that require
unavailable local hardware are skipped under their existing markers.

- [ ] **Step 3: Inspect the final branch state**

Run:

```bash
git diff --check origin/codex/fsdp2-training-track...HEAD
git status --short --branch
git log --oneline --decorate -6
```

Expected: no whitespace errors, no uncommitted files, and only the approved
design/plan/pilot commits ahead of the remote branch.

- [ ] **Step 4: Preview the exact push**

Show the target branch, commit range, complete diffstat, absence of force, no
GPU cost, and the exact command:

```bash
git push origin codex/fsdp2-training-track
```

Stop and wait for `APPROVE PUSH`. After approval, push and verify the local and
remote commit IDs match.

---

### Task 5: Prepare and Run the Two-A40 Pilot Under the External-Action Gate

**Files:**
- Produce on Great Lakes: `results/raw/fsdp_scaling/a40-pilot-$PQS_PILOT_COMMIT/qwen2.5-0.5b/`
- Produce on Great Lakes: `results/raw/fsdp_scaling/a40-pilot-$PQS_PILOT_COMMIT/qwen3-8b/`
- Produce on Great Lakes: `logs/%x-%j.out`
- Produce on Great Lakes: `logs/%x-%j.err`

**Interfaces:**
- Consumes: the pushed implementation commit, `scripts/activate_great_lakes.sh`, the committed SFT gate, cached Hugging Face artifacts, and one homogeneous two-A40 Slurm allocation.
- Produces: raw pilot run records and logs; no curated claim is produced until Task 6 validates them.

- [ ] **Step 1: Pull and perform read-only Great Lakes preflight**

On Great Lakes, fast-forward the isolated checkout to the exact pushed branch,
confirm `git status --short` is empty, source the existing environment, and
resolve both model revisions. Record the full immutable 40-character revision
printed by `resolve_sft_source()` for each model. Confirm the SFT gate is tracked
and unchanged at HEAD.

Set the task-specific path variables from that clean checkout:

```bash
PQS_PILOT_COMMIT="$(git rev-parse --short=7 HEAD)"
PQS_PILOT_ROOT="results/raw/fsdp_scaling/a40-pilot-$PQS_PILOT_COMMIT"
```

Do not submit a job during this step.

- [ ] **Step 2: Dry-run both controllers inside a two-GPU shell context**

Run the launcher with `--dry-run` for these exact experiment shapes:

```text
small model: pilot, world_sizes=1,2, global_batch=8, sequence_length=512,
             micro_batch=1, warmup=3, measure=10, profile=3
Qwen3-8B:   pilot, world_sizes=2,   global_batch=8, sequence_length=2048,
             micro_batch=1, warmup=3, measure=10, profile=3,
             activation checkpointing enabled
```

Use the immutable revisions resolved in Step 1. Confirm the printed commands
contain the correct output directories and never request world sizes 4 or 8.

- [ ] **Step 3: Present the exact protected Slurm preview**

Before submission, show:

- action: one `sbatch` submission running the two controllers sequentially;
- target: Great Lakes account `huterer0`, partition `spgpu`, one node, two A40 GPUs;
- payload: the complete final `sbatch` command with both resolved model revision
  hashes and the pushed Git commit embedded in output paths;
- cost: at most two GPUs for one hour, or two GPU-hours; and
- side effects: allocation consumption, model/dataset cache reads or downloads,
  raw result files, and Slurm logs.

Stop. Submit only after the user replies `APPROVE RUN` to that exact preview.

- [ ] **Step 4: Monitor without changing the experiment**

After submission, monitor `squeue`, then use `sacct` to record state, exit code,
elapsed time, and node. Inspect the job log from the repository directory, not
the login-shell home directory. If any process fails, preserve completed
outputs and diagnose before proposing a retry. Do not automatically resubmit.

---

### Task 6: Validate, Curate, and Document Only the Measurements That Exist

**Files:**
- Create after successful measurement: `results/fsdp_scaling/a40_pilot_qwen2.5_0.5b.json`
- Create after successful measurement: `results/fsdp_scaling/a40_pilot_qwen3_8b_w2.json`
- Create or modify only after successful measurement: `docs/memory_ledger.md`
- Create or modify only after successful measurement: `docs/fsdp_notes.md`

**Interfaces:**
- Consumes: raw top-level `scaling.jsonl`, rank records, run configs, worker results, Slurm accounting, and logs from Task 5.
- Produces: compact committed evidence records and documentation whose empirical claims cite their run IDs.

- [ ] **Step 1: Validate both raw result directories**

Run the validator separately for both controller output directories:

```bash
python -m bench.scaling --validate "$PQS_PILOT_ROOT/qwen2.5-0.5b"
python -m bench.scaling --validate "$PQS_PILOT_ROOT/qwen3-8b"
```

Confirm:

```text
small model: benchmark_mode=pilot, world_sizes=[1,2], 10 durations per rank
Qwen3-8B:   benchmark_mode=pilot, world_sizes=[2], scaling_efficiency=null,
            10 durations on both ranks
```

Reject the result if hardware names differ, a record is dirty, the Qwen3 model
revision is not immutable, any metric is non-finite, or any rank is absent.

- [ ] **Step 2: Create compact evidence records**

Copy only validated controller records and the minimal Slurm provenance needed
to reproduce them. Exclude profiler traces, checkpoints, caches, and raw logs.
Validate each JSON file with `python -m json.tool` and ensure each cited run ID
appears exactly once.

- [ ] **Step 3: Write the memory ledger without filling missing cells**

In `docs/memory_ledger.md`, state the analytical Qwen3-8B predictions first.
Place the measured world-size-2 allocated and reserved values next to the
38.22/41.28 GiB predictions and cite the committed 8B run ID. Mark world sizes
1, 4, and 8 `unmeasured on available A40 allocation`. Explain allocator
residuals only from recorded component counters; do not invent explanations
that are not supported by the record.

- [ ] **Step 4: Write the pilot scaling notes with the strong-scaling warning**

In `docs/fsdp_notes.md`, report the 0.5B world-size-1 and world-size-2
throughputs, the computed efficiency, step spread, MFU, memory, and
communication fractions with run IDs. State immediately that fixed global
batch 8 halves the per-GPU work, making this a communication-heavy
strong-scaling lower bound. State that 0.5B/sequence-512 and
8B/sequence-2048 are separate experiments and cannot be compared.

Do not call the two-point 0.5B pilot the Qwen3 scaling curve. Leave the
production 1/2/4/8 curve and checkpointing ablation explicitly pending.

- [ ] **Step 5: Verify evidence-to-claim traceability**

Run focused JSON/record validation, `git diff --check`, and the full pytest
suite. Search both docs for every empirical number and confirm the adjacent run
ID exists in one of the two curated JSON records.

- [ ] **Step 6: Preview evidence and documentation commits**

Show the exact files, diffstat, run IDs, measured values, and commit command.
Proceed only after a new `APPROVE RUN`. Preview any subsequent remote push
separately and require `APPROVE PUSH`.
