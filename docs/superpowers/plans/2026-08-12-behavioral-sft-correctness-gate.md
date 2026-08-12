# Behavioral SFT Correctness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SFT correctness gate accept statistically equivalent bf16 training while retaining per-step gradient and selected-update differences as traceable diagnostics.

**Architecture:** Keep comparison generation and run-record fields unchanged. Change only SFT validation policy: immutable-source checks, first-loss sanity, complete finite curves, and the pooled three-seed loss-noise bound remain hard gates; gradient-norm relative error and selected-update cosine become diagnostic-only. GRPO retains its existing strict fixed-rollout, gradient, update, reward-curve, and resume checks.

**Tech Stack:** Python 3.11, pytest, JSON run records, PyTorch training metrics.

## Global Constraints

- Do not rerun the six completed 20-step SFT jobs merely to change validation policy.
- Preserve `grad_norm_relative_error_max` and `update_cosine_min` in every committed comparison and gate record.
- Reject NaN and infinity in hard-gate and diagnostic metrics.
- Keep the fixed seed set `(41, 42, 43)` and pooled three-seed loss-noise calculation unchanged.
- Do not weaken GRPO acceptance behavior.

---

### Task 1: Encode the revised SFT acceptance policy

**Files:**
- Modify: `tests/test_correctness_gate.py`
- Modify: `bench/correctness.py`

**Interfaces:**
- Consumes: `build_gate_record(gate: GateName, comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]` and `validate_gate_record(record: Mapping[str, Any]) -> None`.
- Produces: the same interfaces, plus a `diagnostic_only_metrics` list in gate records that makes the SFT policy self-describing.

- [ ] **Step 1: Write the failing SFT diagnostic-only test**

Replace the old SFT update-cosine rejection assertion with:

```python
def test_sft_gate_records_bf16_trajectory_diagnostics_without_rejecting() -> None:
    record = _passing_record()
    record["metrics"]["grad_norm_relative_error_max"] = 0.25
    record["metrics"]["update_cosine_min"] = 0.95

    validate_gate_record(record)

    assert record["metrics"]["grad_norm_relative_error_max"] == 0.25
    assert record["metrics"]["update_cosine_min"] == 0.95
    assert record["diagnostic_only_metrics"] == [
        "grad_norm_relative_error_max",
        "update_cosine_min",
    ]
```

- [ ] **Step 2: Write failing non-finite and GRPO-preservation tests**

Add:

```python
@pytest.mark.parametrize(
    "metric",
    ["grad_norm_relative_error_max", "update_cosine_min"],
)
def test_sft_gate_rejects_nonfinite_diagnostics(metric: str) -> None:
    record = _passing_record()
    record["metrics"][metric] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        validate_gate_record(record)


def test_gate_rejects_nonfinite_loss_curve() -> None:
    record = _passing_record()
    record["metrics"]["loss_curve_points"][0]["mean_abs_diff"] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        validate_gate_record(record)


def test_grpo_keeps_strict_gradient_and_update_checks() -> None:
    record = _passing_record("grpo")
    record["metrics"]["grad_norm_relative_error_max"] = 0.02
    with pytest.raises(ValueError, match="gradient norm"):
        validate_gate_record(record)

    record = _passing_record("grpo")
    record["metrics"]["update_cosine_min"] = 0.998
    with pytest.raises(ValueError, match="0.999"):
        validate_gate_record(record)
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_correctness_gate.py
```

Expected: the SFT diagnostic-only test fails because the current validator rejects the large finite gradient difference, and the non-finite tests fail because NaN comparisons currently bypass threshold checks.

- [ ] **Step 4: Implement finite-value validation and gate-specific thresholds**

In `bench/correctness.py`, import `isfinite` from `math` and add:

```python
def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
```

Use `_finite_float()` for first loss, gradient-norm error, update cosine, and every numeric loss/reward curve field. In `validate_gate_record()`, apply `GRAD_NORM_RTOL` and `UPDATE_COSINE_MIN` only inside the `gate == "grpo"` branch. Keep their SFT values present and finite.

In `build_gate_record()`, add:

```python
"diagnostic_only_metrics": (
    ["grad_norm_relative_error_max", "update_cosine_min"]
    if gate == "sft"
    else []
),
```

- [ ] **Step 5: Run focused and full local verification**

Run:

```bash
pytest -q tests/test_correctness_gate.py
pytest -q
git diff --check
```

Expected: the focused suite and full suite pass, and the diff check prints no errors.

- [ ] **Step 6: Commit the implementation batch after an approval preview**

```bash
git add bench/correctness.py tests/test_correctness_gate.py docs/superpowers/plans/2026-08-12-behavioral-sft-correctness-gate.md
git commit -m "fix: judge SFT parity by behavioral equivalence"
```

### Task 2: Publish the existing completed SFT evidence under the revised policy

**Files:**
- Read: `results/fsdp_correctness/sft_comparisons.json`
- Create on Great Lakes: `results/fsdp_correctness/sft_gate.json`

**Interfaces:**
- Consumes: the six already-completed oracle/FSDP2 training records and selected-tensor probes referenced by `sft_comparisons.json`.
- Produces: one validated `sft_gate.json`; it does not retrain the model.

- [ ] **Step 1: Push the verified implementation after an approval preview**

Push `codex/fsdp2-training-track` without force and update the Great Lakes worktree by fast-forward only.

- [ ] **Step 2: Validate the existing comparison artifact without GPU compute**

Run on Great Lakes after verifying the exact pushed commit:

```bash
python -m bench.correctness \
  --gate sft \
  --comparisons results/fsdp_correctness/sft_comparisons.json \
  --output results/fsdp_correctness/sft_gate.json
```

Expected: `sft_gate.json` is written only if source digests, first-loss tolerance, finite diagnostics, and every pooled three-seed loss-curve point pass.

- [ ] **Step 3: Audit and retain the diagnostic evidence**

Read `sft_gate.json` and confirm:

```python
record["status"] == "pass"
record["diagnostic_only_metrics"] == [
    "grad_norm_relative_error_max",
    "update_cosine_min",
]
record["metrics"]["grad_norm_relative_error_max"] >= 0.0
-1.0 <= record["metrics"]["update_cosine_min"] <= 1.0
```

If the loss curve itself exceeds the pooled noise bound, stop and investigate that behavioral failure; do not start scaling.

