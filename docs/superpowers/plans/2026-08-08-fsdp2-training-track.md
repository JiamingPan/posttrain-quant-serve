# FSDP2 Training Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add direct PyTorch FSDP2 full-parameter SFT and GRPO for Qwen3-8B, with sharded DCP checkpoints, Slurm/torchrun launch, measured scaling records, and unchanged handoff to the repository's Hugging Face quantization and vLLM serving paths.

**Architecture:** A small `train` package owns shared data, FSDP2, checkpoint, memory, and run-record primitives; the required SFT and GRPO entry points compose those primitives without DeepSpeed or an Accelerate wrapper. Correctness runs gate the expensive Qwen3 sweep, DCP consolidation produces a normal sharded-safetensors Hugging Face directory, and documentation is rendered only from committed machine-readable run records.

**Tech Stack:** Python 3.11, PyTorch 2.8.0 cu128, `torch.distributed.fsdp.fully_shard`, DTensor, `torch.distributed.checkpoint`, Transformers 4.51+, TRL 1.5.0 as the GRPO oracle, Hugging Face Datasets, safetensors, pytest, torchrun, Slurm, NCCL, and PyTorch Profiler.

## Global Constraints

- Use PyTorch FSDP2 `torch.distributed.fsdp.fully_shard`; never import or instantiate the legacy `FullyShardedDataParallel` wrapper.
- Do not use DeepSpeed or place Accelerate around FSDP2 calls.
- Preserve the repository's script-driven `argparse` configuration, Slurm environment-to-CLI translation, printed resolved configuration, and append-only JSON/JSONL/CSV tracking style.
- Pin the cluster training runtime to PyTorch `2.8.0` cu128 and require `transformers>=4.51.0`, the version recorded by the Qwen3-8B model configuration.
- Use `Qwen/Qwen3-8B` with revision resolved and recorded at launch; its checked configuration has 36 decoder blocks, hidden size 4096, intermediate size 12288, 8 KV heads, and 151,936 vocabulary entries.
- Use bf16 resident parameters, bf16 gradients, bf16 AdamW moments, bf16 parameter all-gathers, fp32 gradient reduction, and bf16 outputs for the homogeneous 1/2/4/8 scaling curve.
- Keep activation checkpointing configurable and enabled by default; use non-reentrant block checkpointing with RNG preservation and `use_cache=False`.
- Reduce-scatter each accumulation microbatch by default; expose unsynchronized accumulation only as an explicit ablation.
- Save DCP only at optimizer-step boundaries and publish a checkpoint only after every rank completes the write.
- Never begin Qwen3 scaling until both one-GPU correctness records pass and are committed.
- Never write measured claims into `docs/memory_ledger.md` or `docs/fsdp_notes.md` unless the cited run ID exists in a committed JSON record.
- Preserve `scripts/train_grpo_gsm8k.py` as the GRPO comparison path and keep existing quantization, evaluation, benchmark, and serving interfaces unchanged.
- Work in an isolated `codex/` worktree when execution begins; preserve the user's unrelated dirty files in the current checkout.

---

## File Map

### Shared training package

- `train/__init__.py`: package marker only.
- `train/run_tracking.py`: resolved configuration, environment identity, JSONL/CSV writes, rank aggregation, and committed-record validation.
- `train/memory_model.py`: analytical SFT/GRPO memory predictions and VRAM preflight.
- `train/gsm8k_data.py`: SFT examples, fixed-length packing, GRPO prompts, local collation, and checkpointable distributed sampling.
- `train/fsdp_utils.py`: process-group lifecycle, Qwen block checkpointing, bottom-up FSDP2 grouping, sharded model construction, global gradient clipping, and rollout shard-state control.
- `train/checkpointing.py`: Hugging Face safetensor-to-DTensor loading, DCP save/resume, progress/RNG state, success markers, and latest-checkpoint resolution.
- `train/grpo_core.py`: rollout representation, grouped rewards/advantages, token log-probabilities, Dr. GRPO objective, reference KL, and policy rollout generation.
- `train/fsdp_sft.py`: required SFT CLI and training loop; also exposes the real SFT optimizer-step function used by parity and scaling.
- `train/fsdp_grpo.py`: required GRPO CLI and loop; records policy/reference/rollout placement and phase peaks.

### Launch, conversion, benchmarks, and reports

- `scripts/consolidate_dcp.py`: distributed model-only DCP conversion into a Hugging Face sharded-safetensors directory.
- `scripts/launch_slurm.sh`: single-node `torchrun` launcher for 1/2/4/8 GPUs and SFT, GRPO, correctness, consolidation, or scaling stages.
- `bench/__init__.py`: package marker only.
- `bench/correctness.py`: SFT and GRPO one-GPU gate runner and validator.
- `bench/scaling.py`: scaling controller/worker, fixed-global-batch measurement, profiler trace accounting, and append-only records.
- `bench/reporting.py`: deterministic generation and provenance validation for the two FSDP documents.
- `results/fsdp_correctness/`: committed gate records.
- `results/fsdp_scaling/`: committed scaling and activation-checkpointing records.
- `docs/memory_ledger.md`: predicted-first, measured-second component memory table.
- `docs/fsdp_notes.md`: scaling efficiency, degradation mechanism, and activation-checkpointing ablation.

### Tests and existing files

- `pytest.ini`: fast, CUDA, distributed, parity, and Slurm marker definitions.
- `requirements-test.txt`: test-only dependencies.
- `tests/__init__.py`, `tests/conftest.py`, and `tests/tiny_qwen.py`: importable test helpers, a torchrun-result fixture, and a CLI that writes a deterministic tiny local Qwen3 checkpoint.
- `tests/test_run_tracking.py`, `tests/test_memory_model.py`, `tests/test_gsm8k_data.py`, `tests/test_grpo_core.py`, `tests/test_scaling_metrics.py`: CPU tests.
- `tests/test_fsdp_distributed.py`, `tests/test_checkpointing_distributed.py`, `tests/test_consolidate_dcp.py`: CUDA/torchrun integration tests using a tiny Qwen-shaped model.
- `tests/test_cli_contracts.py`, `tests/test_launch_slurm.py`, `tests/test_report_provenance.py`: CLI and artifact contract tests.
- Modify `requirements.txt:6` to raise the Transformers floor.
- Modify `configs/README.md:1-15` to document the new script-driven FSDP2 path.
- Modify `scripts/train_grpo_gsm8k.py:55-174` only to expose deterministic seed and optional run-record output for the oracle; its defaults and training behavior remain unchanged.

---

### Task 1: Runtime Floors, Package Skeleton, and Test Markers

**Files:**
- Create: `train/__init__.py`
- Create: `bench/__init__.py`
- Create: `tests/__init__.py`
- Create: `requirements-test.txt`
- Create: `pytest.ini`
- Create: `tests/test_runtime_contract.py`
- Modify: `requirements.txt:1-13`
- Modify: `configs/README.md:1-15`

**Interfaces:**
- Consumes: the existing Python 3.11/cu128 environment instructions.
- Produces: importable `train` and `bench` packages, pytest markers, and explicit PyTorch/Transformers compatibility floors used by every later task.

- [ ] **Step 1: Write the failing runtime-contract test**

```python
from pathlib import Path


def test_qwen3_transformers_floor_and_no_legacy_fsdp_dependency():
    requirements = Path("requirements.txt").read_text()
    assert "transformers>=4.51.0" in requirements
    assert "#   torch==2.8.0" in requirements
    assert "FullyShardedDataParallel" not in requirements
```

- [ ] **Step 2: Run the test and verify the current floor fails**

Run: `pytest tests/test_runtime_contract.py -v`

Expected: FAIL because `requirements.txt` currently declares `transformers>=4.45.0` and documents PyTorch only in comments.

- [ ] **Step 3: Establish the runtime and test files**

Keep PyTorch as a separately installed CUDA wheel and retain this exact documented pin:

```text
#   torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

Raise the installed Transformers requirement to:

```text
transformers>=4.51.0
```

Use these test dependencies:

```text
pytest>=8.0.0
```

Use these pytest markers:

```ini
[pytest]
markers =
    cuda: requires at least one CUDA GPU
    distributed: launches more than one torchrun rank
    parity: compares independent training paths
    slurm: requires a Slurm allocation
```

Update `configs/README.md` to retain the script-driven convention and name `python -m train.fsdp_sft`, `python -m train.fsdp_grpo`, and `scripts/launch_slurm.sh` as the new entry points.

- [ ] **Step 4: Run the fast contract test**

Run: `pytest tests/test_runtime_contract.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the runtime contract**

```bash
git add train/__init__.py bench/__init__.py tests/__init__.py requirements-test.txt pytest.ini tests/test_runtime_contract.py requirements.txt configs/README.md
git commit -m "build: establish FSDP2 runtime contract"
```

---

### Task 2: Run Records and Analytical Memory Preflight

**Files:**
- Create: `train/run_tracking.py`
- Create: `train/memory_model.py`
- Create: `tests/test_run_tracking.py`
- Create: `tests/test_memory_model.py`

**Interfaces:**
- Consumes: `argparse.Namespace`, process environment, CUDA device properties, and resolved model configuration.
- Produces: `RunIdentity`, `MemoryPrediction`, `capture_run_identity()`, `append_run_record()`, `gather_rank_records()`, `predict_sft_peak()`, `predict_grpo_peak()`, and `assert_memory_fits()`.

- [ ] **Step 1: Write failing tests for record append behavior and the approved memory table**

```python
def test_native_bf16_predictions_match_design():
    expected = {1: 66.03, 2: 38.22, 4: 22.96, 8: 15.33}
    for world_size, allocated_gib in expected.items():
        row = predict_sft_peak(8_190_735_360, world_size, checkpointing=True)
        assert row.allocated_gib == pytest.approx(allocated_gib, abs=0.02)
        assert row.reserved_gib == pytest.approx(allocated_gib * 1.08, abs=0.05)


def test_append_record_writes_jsonl_and_stable_csv_columns(tmp_path):
    row = {"run_id": "sft-w1-seed42", "world_size": 1, "tokens_per_sec": 10.0}
    append_run_record(tmp_path, "scaling", row, tuple(row))
    assert json.loads((tmp_path / "scaling.jsonl").read_text().strip()) == row
    assert (tmp_path / "scaling.csv").read_text().splitlines()[0] == "run_id,world_size,tokens_per_sec"
```

- [ ] **Step 2: Run the tests and verify missing modules fail**

Run: `pytest tests/test_run_tracking.py tests/test_memory_model.py -v`

Expected: collection errors for `train.run_tracking` and `train.memory_model`.

- [ ] **Step 3: Implement immutable identities and append-only writes**

Define these public types and functions:

```python
@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    stage: str
    git_commit: str
    git_dirty: bool
    slurm_job_id: str | None
    hostname: str
    started_at_utc: str


def capture_run_identity(stage: str, config: Mapping[str, object]) -> RunIdentity:
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RunIdentity(
        run_id=f"{stage}-{started}-{config_hash}",
        stage=stage,
        git_commit=_git_output("rev-parse", "HEAD"),
        git_dirty=bool(_git_output("status", "--porcelain")),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=socket.gethostname(),
        started_at_utc=started,
    )
```

`append_run_record()` must use `json.dumps(..., sort_keys=True)`, retain the repository's JSONL-plus-CSV layout, and reject a CSV schema change instead of silently reordering columns.

- [ ] **Step 4: Implement formula-first memory accounting**

```python
@dataclass(frozen=True)
class MemoryPrediction:
    params_gib: float
    grads_gib: float
    optimizer_gib: float
    activations_gib: float
    collectives_gib: float
    other_gib: float
    allocated_gib: float
    reserved_gib: float


def predict_sft_peak(parameter_count: int, world_size: int, checkpointing: bool) -> MemoryPrediction:
    shard = parameter_count * 2 / 1024**3 / world_size
    params, grads, optimizer = shard, shard, 2 * shard
    activations = 3.5 if checkpointing else 13.5
    collectives = 0.0 if world_size == 1 else 2.7
    other = 1.5
    allocated = params + grads + optimizer + activations + collectives + other
    return MemoryPrediction(params, grads, optimizer, activations, collectives, other, allocated, allocated * 1.08)
```

`predict_grpo_peak()` must separately return training and rollout phase peaks, add no reference allocation for `beta=0`, add one bf16 reference shard for `beta>0`, and account for the full 15.26 GiB all-gather output only in `keep_unsharded` rollout mode.

`assert_memory_fits()` must raise before model construction when predicted reserved memory exceeds 95% of detected VRAM and include world size, state precision, checkpointing, phase, prediction, and detected capacity in the error.

- [ ] **Step 5: Run the pure tests**

Run: `pytest tests/test_run_tracking.py tests/test_memory_model.py -v`

Expected: PASS, including the 1/2/4/8 analytical rows and activation-checkpointing-off rows.

- [ ] **Step 6: Commit record and memory primitives**

```bash
git add train/run_tracking.py train/memory_model.py tests/test_run_tracking.py tests/test_memory_model.py
git commit -m "feat: add FSDP run records and memory model"
```

---

### Task 3: GSM8K SFT Data, Packing, and Resume-Safe Sampling

**Files:**
- Create: `train/gsm8k_data.py`
- Create: `tests/test_gsm8k_data.py`

**Interfaces:**
- Consumes: a Hugging Face tokenizer and GSM8K rows containing `question` and `answer`.
- Produces: `SFTFeature`, `TokenBatch`, `build_sft_feature()`, `pack_sft_features()`, `build_grpo_prompt()`, `collate_token_batches()`, and `CheckpointableDistributedSampler`.

- [ ] **Step 1: Write failing assistant-mask, packing, and sampler-resume tests**

```python
def test_only_assistant_suffix_is_supervised(fake_qwen_tokenizer):
    feature = build_sft_feature(fake_qwen_tokenizer, "2+2?", "reasoning\n#### 4", max_length=64)
    first_label = next(i for i, value in enumerate(feature.labels) if value != -100)
    assert feature.labels[:first_label] == [-100] * first_label
    assert feature.labels[first_label:] == feature.input_ids[first_label:]


def test_sampler_resume_returns_exact_next_local_indices():
    sampler = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7, shuffle=True)
    first = [sampler.next_indices(2) for _ in range(3)]
    state = sampler.state_dict()
    expected = sampler.next_indices(2)
    resumed = CheckpointableDistributedSampler(16, rank=1, world_size=2, seed=7, shuffle=True)
    resumed.load_state_dict(state)
    assert resumed.next_indices(2) == expected
    assert len(first) == 3
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run: `pytest tests/test_gsm8k_data.py -v`

Expected: collection error for `train.gsm8k_data`.

- [ ] **Step 3: Implement prefix-checked assistant-only tokenization**

```python
def build_sft_feature(tokenizer, question: str, answer: str, max_length: int) -> SFTFeature:
    user = [{"role": "user", "content": gsm8k_user_text(question)}]
    conversation = [*user, {"role": "assistant", "content": answer}]
    prompt_ids = tokenizer.apply_chat_template(user, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("The chat template is not prefix-preserving for assistant-only SFT labels")
    input_ids = full_ids[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    return SFTFeature(input_ids=input_ids, labels=labels)
```

Build `gsm8k_user_text()` from the exact instruction already used by `scripts/gsm8k_reward.py:29-40`; import or reuse it rather than maintaining different wording.

- [ ] **Step 4: Implement deterministic packing and sampling**

`pack_sft_features()` must concatenate examples in deterministic input order, insert EOS between examples, retain `-100` for all user/padding positions, emit exactly `sequence_length` tokens per packed row, and drop or pad the final row according to an explicit flag. `CheckpointableDistributedSampler` must save `epoch` and global sample cursor and derive rank-local slices from one global permutation, so each optimizer-step checkpoint resumes at the exact next example.

- [ ] **Step 5: Run all data tests**

Run: `pytest tests/test_gsm8k_data.py -v`

Expected: PASS for assistant masking, prefix rejection, fixed-length packing, rank disjointness, epoch rollover, and exact resume.

- [ ] **Step 6: Commit the data path**

```bash
git add train/gsm8k_data.py tests/test_gsm8k_data.py
git commit -m "feat: add deterministic GSM8K SFT data path"
```

---

### Task 4: Direct FSDP2 Construction, Checkpoint Wrapping, and Global Clipping

**Files:**
- Create: `train/fsdp_utils.py`
- Create: `tests/conftest.py`
- Create: `tests/tiny_qwen.py`
- Create: `tests/test_fsdp_distributed.py`
- Create: `tests/workers/fsdp_layout_worker.py`

**Interfaces:**
- Consumes: a meta-initialized Qwen causal LM, torchrun environment variables, and `FSDPSettings`.
- Produces: `DistContext`, `FSDPSettings`, `init_distributed()`, `destroy_distributed()`, `apply_qwen_activation_checkpointing()`, `fully_shard_qwen()`, `fsdp_modules()`, `clip_global_grad_norm_()`, and `rollout_parameter_state()`.

- [ ] **Step 1: Write a two-rank failing sharding-layout test**

```python
@pytest.mark.cuda
@pytest.mark.distributed
def test_qwen_groups_are_explicit_and_parameters_are_dtensors(torchrun_result):
    payload = torchrun_result("tests/workers/fsdp_layout_worker.py", nproc=2)
    assert payload["groups"] == ["embed_tokens", "layer.0", "layer.1", "lm_head", "root"]
    assert payload["all_parameters_dtensor_after_forward"] is True
    assert payload["resident_numel_sum"] == payload["global_numel"]
```

`tests/tiny_qwen.py` must expose `build_tiny_qwen3()` and a CLI that deterministically writes a local model/tokenizer directory. The worker must use its `Qwen3Config` with two layers, hidden size 64, intermediate size 128, eight heads, two KV heads, vocabulary 128, and no network download.

- [ ] **Step 2: Run the worker test and verify the shared FSDP module is missing**

Run: `pytest tests/test_fsdp_distributed.py::test_qwen_groups_are_explicit_and_parameters_are_dtensors -v`

Expected: FAIL during import of `train.fsdp_utils`.

- [ ] **Step 3: Implement the reusable torchrun fixture and tiny Qwen writer**

Add one reusable CUDA fixture that launches workers and reads their rank-zero JSON result:

```python
@pytest.fixture
def torchrun_result(tmp_path):
    def run(worker: str, nproc: int, **worker_args: object) -> dict[str, object]:
        result_path = tmp_path / f"{Path(worker).stem}-{nproc}.json"
        cmd = [
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            f"--nproc-per-node={nproc}", worker, "--result-path", str(result_path),
        ]
        for key, value in worker_args.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        subprocess.run(cmd, check=True)
        return json.loads(result_path.read_text())
    return run
```

Implement `tests/tiny_qwen.py` with the exact tiny configuration stated in Step 1 and deterministic `save_pretrained()` output. Build its local tokenizer without a download:

```python
vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
vocab.update({f"tok{index}": index + 4 for index in range(124)})
backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
backend.pre_tokenizer = Whitespace()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend,
    pad_token="<pad>", unk_token="<unk>", bos_token="<bos>", eos_token="<eos>",
)
tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }} <eos> {% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
torch.manual_seed(0)
model = Qwen3ForCausalLM(tiny_qwen3_config())
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)
```

- [ ] **Step 4: Implement distributed initialization and bottom-up grouping**

```python
@dataclass(frozen=True)
class DistContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    mesh: DeviceMesh


def fully_shard_qwen(model: nn.Module, ctx: DistContext, settings: FSDPSettings) -> list[str]:
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    groups: list[str] = []
    fully_shard(model.model.embed_tokens, mesh=ctx.mesh, reshard_after_forward=True, mp_policy=policy)
    groups.append("embed_tokens")
    for index, block in enumerate(model.model.layers):
        fully_shard(block, mesh=ctx.mesh, reshard_after_forward=True, mp_policy=policy)
        groups.append(f"layer.{index}")
    fully_shard(model.lm_head, mesh=ctx.mesh, reshard_after_forward=True, mp_policy=policy)
    groups.append("lm_head")
    fully_shard(model, mesh=ctx.mesh, reshard_after_forward=True, mp_policy=policy)
    groups.append("root")
    return groups
```

`init_distributed()` must read `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`, bind the CUDA device before `init_process_group("nccl")`, use a configurable timeout, initialize a one-dimensional mesh named `dp`, and reject non-CUDA execution for the training entry points.

- [ ] **Step 5: Implement configurable non-reentrant block checkpointing**

Replace each element of `model.model.layers` with `checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=True)` only when enabled. Set `model.config.use_cache=False` for training and retain a test that disabled checkpointing leaves block identities unchanged.

- [ ] **Step 6: Implement DTensor-aware clipping and rollout state control**

```python
def clip_global_grad_norm_(model: nn.Module, max_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm, foreach=False)
```

Call it on every rank. `rollout_parameter_state(model, "keep_unsharded")` must set `reshard_after_forward=False` on each FSDP group, unshard every group, and in `finally` reshard every group and restore `reshard_after_forward=True`. `"reshard"` mode must leave normal per-forward gathering active.

- [ ] **Step 7: Run the two-rank layout, checkpoint flag, clipping, and cleanup tests**

Run: `pytest tests/test_fsdp_distributed.py -v`

Expected: PASS; the norm must equal an unsharded reference norm, and both ranks must return the same scalar.

- [ ] **Step 8: Commit the FSDP2 core**

```bash
git add train/fsdp_utils.py tests/conftest.py tests/tiny_qwen.py tests/test_fsdp_distributed.py tests/workers/fsdp_layout_worker.py
git commit -m "feat: add direct Qwen FSDP2 sharding core"
```

---

### Task 5: Hugging Face Sharded Load and Atomic DCP Save/Resume

**Files:**
- Create: `train/checkpointing.py`
- Create: `tests/test_checkpointing_distributed.py`
- Create: `tests/workers/checkpoint_worker.py`

**Interfaces:**
- Consumes: an already-sharded/meta-materialized model, optimizer created from DTensor parameters, scheduler, sampler, resolved config, and DCP/Hugging Face checkpoint paths.
- Produces: `TrainProgress`, `load_hf_weights_into_shards()`, `save_dcp_checkpoint()`, `load_dcp_checkpoint()`, `resolve_resume_checkpoint()`, and `checkpoint_manifest()`.

- [ ] **Step 1: Write a failing two-rank uninterrupted-versus-resumed test**

```python
@pytest.mark.cuda
@pytest.mark.distributed
def test_dcp_resume_restores_exact_next_step(torchrun_result, tmp_path):
    row = torchrun_result("tests/workers/checkpoint_worker.py", nproc=2, output_dir=tmp_path)
    assert row["success_marker_present"] is True
    assert row["partial_directory_selected"] is False
    assert row["next_loss_resumed"] == pytest.approx(row["next_loss_uninterrupted"], abs=1e-6)
    assert row["optimizer_state_keys_before"] == row["optimizer_state_keys_after"]
    assert row["sampler_cursor_before"] == row["sampler_cursor_after"]
```

- [ ] **Step 2: Run the test and verify the checkpoint module is missing**

Run: `pytest tests/test_checkpointing_distributed.py -v`

Expected: FAIL during import of `train.checkpointing`.

- [ ] **Step 3: Implement meta-to-shard Hugging Face loading**

Use this ordering:

```python
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=resident_dtype)
apply_qwen_activation_checkpointing(model, enabled=activation_checkpointing)
fully_shard_qwen(model, ctx, fsdp_settings)
model.to_empty(device=ctx.device)
model_state = get_model_state_dict(model)
dcp.load(model_state, storage_reader=HuggingFaceStorageReader(snapshot_path))
incompatible = set_model_state_dict(model, model_state, options=StateDictOptions(strict=True))
```

Resolve model IDs through `huggingface_hub.snapshot_download()` and record the immutable snapshot commit. Never call `AutoModelForCausalLM.from_pretrained()` on each GPU for the 8B training load.

- [ ] **Step 4: Implement checkpoint state and exact resume ordering**

```python
@dataclass
class TrainProgress:
    global_step: int
    consumed_tokens: int
    sampler_state: dict[str, int]
    rng_states: list[dict[str, object]]
    config: dict[str, object]
    source_digests: dict[str, str]
```

Before save, gather Python, NumPy, CPU Torch, and CUDA RNG states from every rank with `dist.all_gather_object()`. Save model and optimizer tensors from `get_state_dict(model, optimizer)` plus scheduler and `TrainProgress`. On load, create the optimizer after `fully_shard_qwen()`, obtain destination state dictionaries with `get_state_dict()`, call `dcp.load()`, then `set_state_dict()` before any backward. Restore scheduler, sampler, progress, and the current rank's RNG state, then validate optimizer FQN count, tensor shape, dtype, global step, and source digests.

- [ ] **Step 5: Implement safe publication**

Generate one UUID on rank 0 and broadcast it. Write to `<output>/.step-00000010-<uuid>.tmp`, synchronize, write `manifest.json` and `_SUCCESS` on rank 0, atomically rename to `<output>/step-00000010`, and synchronize again. `resolve_resume_checkpoint("latest")` must ignore directories without `_SUCCESS` and sort by numeric optimizer step.

- [ ] **Step 6: Run save/resume and world-size-change tests**

Run: `pytest tests/test_checkpointing_distributed.py -v`

Expected: PASS for exact same-world-size continuation, ignored partial writes, optimizer-state restoration, and model/optimizer resharding at a different world size marked `bitwise_resume=false`.

- [ ] **Step 7: Commit checkpoint support**

```bash
git add train/checkpointing.py tests/test_checkpointing_distributed.py tests/workers/checkpoint_worker.py
git commit -m "feat: add sharded DCP save and resume"
```

---

### Task 6: Full-Parameter SFT Entry Point and Shared Optimizer Step

**Files:**
- Create: `train/fsdp_sft.py`
- Create: `tests/test_sft_step.py`
- Create: `tests/test_cli_contracts.py`

**Interfaces:**
- Consumes: Tasks 2-5 primitives and GSM8K packed batches.
- Produces: `SFTConfig`, `parse_sft_args()`, `prepare_step_batches()`, `sft_optimizer_step()`, `run_sft()`, and CLI `python -m train.fsdp_sft`.

- [ ] **Step 1: Write failing global-token-mean and CLI-default tests**

```python
def test_sft_backward_scale_uses_global_step_token_count():
    scale = backward_scale(world_size=4, global_step_tokens=8192)
    assert scale == pytest.approx(4 / 8192)


def test_sft_cli_defaults_enable_checkpointing():
    args = parse_sft_args(["--output_dir", "/tmp/run"])
    assert args.model == "Qwen/Qwen3-8B"
    assert args.sharding == "fsdp2"
    assert args.activation_checkpointing is True
    assert args.gradient_accumulation_steps == 8
    assert args.max_grad_norm == 1.0
```

- [ ] **Step 2: Run the tests and verify the entry point is missing**

Run: `pytest tests/test_sft_step.py tests/test_cli_contracts.py::test_sft_cli_defaults_enable_checkpointing -v`

Expected: collection error for `train.fsdp_sft`.

- [ ] **Step 3: Implement one mathematically correct sharded optimizer step**

Collect the configured number of local microbatches before backward, sum their valid assistant-token counts, and all-reduce that count. For each microbatch:

```python
logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False).logits
shift_logits = logits[:, :-1].float()
shift_labels = batch.labels[:, 1:]
local_loss_sum = F.cross_entropy(
    shift_logits.reshape(-1, shift_logits.size(-1)),
    shift_labels.reshape(-1),
    ignore_index=-100,
    reduction="sum",
)
loss_for_backward = local_loss_sum * ctx.world_size / global_step_tokens
loss_for_backward.backward()
```

The `world_size` factor compensates for FSDP's averaged gradient reduction. After all microbatches, call `clip_global_grad_norm_()` on every rank, then optimizer step, scheduler step, and `zero_grad(set_to_none=True)`. Return globally reduced loss sum/token count, pre-clip norm, clipped flag, useful tokens, elapsed time, and local peak allocated/reserved bytes.

For the default `reduce_scatter` accumulation mode, leave gradient synchronization enabled for every backward so each microbatch accumulates into DTensor gradient shards. Only when the user explicitly selects `no_sync`, call `model.set_requires_gradient_sync(False)` for non-final microbatches and restore `True` before the final backward; record that mode and its full-gradient peak separately.

- [ ] **Step 4: Implement the CLI loop and records**

Create `torch.optim.AdamW(model.parameters(), fused=True)` only after FSDP2 conversion and weight loading. The native-bf16 profile must assert after the first step that every `exp_avg` and `exp_avg_sq` tensor is bf16; the fp32-resident option must assert fp32 instead. Expose `--sharding {fsdp2,none}` with `fsdp2` as the production default and `none` accepted only at world size one for the SFT correctness oracle. Also expose paired `--activation_checkpointing/--no_activation_checkpointing`, `--accumulation_sync {reduce_scatter,no_sync}`, `--resume {none,latest,<path>}`, model/revision, resident precision, split/limit, packing, sequence length, local microbatch, accumulation, max steps, learning rate, weight decay, warmup, max norm, seed, logging/save intervals, and output directory. Print the fully resolved configuration on rank 0, write `run_config.json`, append step records to `train.jsonl`, save DCP only after optimizer steps, and destroy the process group in `finally`.

- [ ] **Step 5: Run CPU and one-/two-GPU SFT step tests**

Run: `pytest tests/test_sft_step.py tests/test_cli_contracts.py -v`

Run on a CUDA host: `pytest tests/test_fsdp_distributed.py tests/test_checkpointing_distributed.py -v`

Expected: PASS; world-size-1 sharded and unsharded gradients agree on the tiny model, two ranks produce the same global token mean with unequal local label counts, and native-bf16 AdamW moment tensors stay bf16.

- [ ] **Step 6: Commit SFT**

```bash
git add train/fsdp_sft.py tests/test_sft_step.py tests/test_cli_contracts.py
git commit -m "feat: add full-parameter FSDP2 SFT"
```

---

### Task 7: GRPO Objective with Fixed-Rollout Oracle

**Files:**
- Create: `train/grpo_core.py`
- Create: `tests/test_grpo_core.py`
- Create: `tests/fixtures/grpo_fixed_rollout.json`

**Interfaces:**
- Consumes: completion log-probabilities, detached old log-probabilities, optional reference log-probabilities, masks, grouped verifiable rewards, clipping bounds, beta, and maximum completion length.
- Produces: `RolloutBatch`, `group_advantages()`, `select_token_logps()`, `grpo_loss_sum()`, and `GRPOMetrics`.

- [ ] **Step 1: Write failing reward/advantage and scalar-objective tests**

```python
def test_scale_rewards_none_centers_without_standardizing():
    rewards = torch.tensor([[1.0, 0.0, -0.25, 0.0]])
    assert torch.equal(group_advantages(rewards, scale_rewards="none"), rewards - rewards.mean(dim=1, keepdim=True))


def test_dr_grpo_matches_independent_scalar_fixture():
    fixture = json.loads(Path("tests/fixtures/grpo_fixed_rollout.json").read_text())
    result = grpo_loss_sum(**tensorize_fixture(fixture))
    assert result.loss_sum.item() == pytest.approx(fixture["expected_loss_sum"], abs=1e-7)
    assert result.kl_sum.item() == pytest.approx(fixture["expected_kl_sum"], abs=1e-7)
```

- [ ] **Step 2: Run the tests and verify the GRPO core is missing**

Run: `pytest tests/test_grpo_core.py -v`

Expected: collection error for `train.grpo_core`.

- [ ] **Step 3: Implement the pinned Dr. GRPO math**

```python
log_ratio = current_logps - old_logps
ratio = log_ratio.exp()
unclipped = ratio * advantages[:, None]
clipped = ratio.clamp(1.0 - epsilon_low, 1.0 + epsilon_high) * advantages[:, None]
policy_loss = -torch.minimum(unclipped, clipped)
if ref_logps is not None:
    delta = ref_logps - current_logps
    per_token_kl = delta.exp() - delta - 1.0
    policy_loss = policy_loss + beta * per_token_kl
masked_loss_sum = (policy_loss * completion_mask).sum()
```

`scale_rewards="none"` subtracts the group mean and does not divide by group standard deviation. Dr. GRPO normalizes the globally summed token loss by `global_completion_count * max_completion_length`; the distributed entry point applies the same `world_size / global_denominator` compensation used by SFT.

- [ ] **Step 4: Reuse the existing exact-match reward unchanged**

The entry point must call `scripts.gsm8k_reward.gsm8k_exact_match_reward`; tests must include clean correct `1.0`, leaky correct `0.75`, clean wrong `0.0`, and leaky wrong `-0.25` cases so the FSDP path cannot drift from the existing reward.

- [ ] **Step 5: Run the fixed-objective suite**

Run: `pytest tests/test_grpo_core.py -v`

Run: `python scripts/check_reward_parser.py`

Expected: PASS for advantage centering, PPO clipping, beta-zero removal of KL, beta-positive KL, completion masks, and Dr. GRPO normalization.

- [ ] **Step 6: Commit the GRPO objective**

```bash
git add train/grpo_core.py tests/test_grpo_core.py tests/fixtures/grpo_fixed_rollout.json
git commit -m "feat: add verifiable Dr GRPO objective"
```

---

### Task 8: Sharded Policy Rollouts and Optional Reference Policy

**Files:**
- Modify: `train/grpo_core.py`
- Modify: `train/fsdp_utils.py`
- Create: `tests/test_grpo_rollout.py`
- Create: `tests/workers/grpo_rollout_worker.py`

**Interfaces:**
- Consumes: an FSDP2 policy, tokenizer, prompt rows, `RolloutConfig`, optional separately sharded frozen reference, and `DistContext`.
- Produces: `RolloutConfig`, `choose_rollout_mode()`, `generate_rollout_batch()`, `teacher_forced_logps()`, and CPU-backed `RolloutBatch` objects.

- [ ] **Step 1: Write failing rollout placement and reshard cleanup tests**

```python
def test_auto_rollout_mode_falls_back_when_full_policy_will_not_fit():
    assert choose_rollout_mode(predicted_keep_unsharded_gib=42.0, capacity_gib=40.0, requested="auto") == "reshard"


@pytest.mark.cuda
@pytest.mark.distributed
def test_keep_unsharded_rollout_reshards_in_finally(torchrun_result):
    row = torchrun_result("tests/workers/grpo_rollout_worker.py", nproc=2)
    assert row["all_groups_sharded_after_success"] is True
    assert row["all_groups_sharded_after_forced_error"] is True
```

- [ ] **Step 2: Run the tests and verify rollout APIs are absent**

Run: `pytest tests/test_grpo_rollout.py -v`

Expected: FAIL because rollout APIs are not defined.

- [ ] **Step 3: Implement synchronized policy generation**

`generate_rollout_batch()` must left-pad prompts, repeat each prompt exactly eight times, call the same policy's `generate()` with temperature 1.0 and configured top-p/top-k, and synchronize completion by using one identical maximum-generation loop on every rank. Pad exhausted local prompt groups so every rank executes identical collectives. Store prompt IDs, completion IDs, attention masks, old policy log-probabilities, rewards, and decoded completion metadata on CPU immediately after each group.

Use `rollout_parameter_state()` around generation. In `reshard` mode, every decoder call gathers and releases layer weights. In `keep_unsharded` mode, preflight the full-policy allocation, unshard all groups once, retain them through autoregressive generation, and reshard in `finally`.

- [ ] **Step 4: Implement sequential teacher-forced reference scoring**

`teacher_forced_logps()` evaluates an optional already-constructed reference in inference mode after policy rollout parameters have been resharded. Tests must supply a frozen tiny reference and verify that it creates no gradients and that reference scoring never overlaps with the keep-unsharded policy context.

- [ ] **Step 5: Run rollout tests**

Run: `pytest tests/test_grpo_rollout.py -v`

Run on two CUDA GPUs: `pytest tests/test_grpo_rollout.py -m distributed -v`

Expected: PASS, including deterministic seeded generations, CPU-backed finished rollouts, frozen supplied-reference parameters, equal collective counts, and cleanup after an injected exception.

- [ ] **Step 6: Commit rollout and reference handling**

```bash
git add train/grpo_core.py train/fsdp_utils.py tests/test_grpo_rollout.py tests/workers/grpo_rollout_worker.py
git commit -m "feat: add FSDP2 GRPO rollout memory modes"
```

---

### Task 9: Full-Parameter GRPO Entry Point

**Files:**
- Create: `train/fsdp_grpo.py`
- Modify: `tests/test_cli_contracts.py`
- Create: `tests/test_grpo_step.py`

**Interfaces:**
- Consumes: an SFT Hugging Face or DCP source, Tasks 2-5 infrastructure, and Tasks 7-8 GRPO primitives.
- Produces: `GRPOConfig`, `parse_grpo_args()`, `grpo_optimizer_step()`, `maybe_build_reference()`, `run_grpo()`, and CLI `python -m train.fsdp_grpo`.

- [ ] **Step 1: Write failing final-recipe defaults and phase-record tests**

```python
def test_grpo_defaults_match_committed_single_gpu_recipe():
    args = parse_grpo_args(["--model", "/sft", "--output_dir", "/run"])
    assert args.num_generations == 8
    assert args.loss_type == "dr_grpo"
    assert args.scale_rewards == "none"
    assert args.beta == 0.0
    assert args.temperature == 1.0


def test_memory_record_names_policy_reference_and_rollout_locations():
    row = build_memory_layout_record(beta=0.0, rollout_mode="reshard")
    assert row["policy"] == "fsdp2_sharded_gpu"
    assert row["reference"] == "absent_beta_zero"
    assert row["rollout_records"] == "cpu_after_group"


def test_beta_zero_does_not_build_reference(monkeypatch):
    monkeypatch.setattr("train.fsdp_grpo.build_reference_model", Mock(side_effect=AssertionError))
    assert maybe_build_reference(beta=0.0, source="sft", ctx=object()) is None
```

- [ ] **Step 2: Run the tests and verify the GRPO entry point is missing**

Run: `pytest tests/test_grpo_step.py tests/test_cli_contracts.py::test_grpo_defaults_match_committed_single_gpu_recipe -v`

Expected: collection error for `train.fsdp_grpo`.

- [ ] **Step 3: Implement phase-separated GRPO updates**

For each optimizer step: generate policy rollouts, compute existing GSM8K rewards, compute old policy log-probabilities, optionally compute reference log-probabilities, move the completed rollout to CPU, reshard both models, then run current-policy teacher forcing in microbatches. Apply `grpo_loss_sum()`, scale by `world_size / (global_completion_count * max_completion_length)`, backward with default per-microbatch reduce-scatter, clip globally on every rank, step AdamW/scheduler, zero gradients, and save only at the complete step boundary.

`maybe_build_reference()` must return `None` at beta zero. At positive beta it loads an independent frozen bf16 FSDP2 copy from the immutable SFT source, creates no optimizer, and records source path, revision, and digest; policy checkpoints contain only that metadata, never duplicate reference weights.

- [ ] **Step 4: Record explicit phase memory and timing**

Reset CUDA peak counters at rollout, optional reference, policy forward/backward, optimizer, and checkpoint phase boundaries. Append phase allocated/reserved peaks, wall times, completion lengths, reward mean/std, zero-variance group fraction, KL, clip ratio, global gradient norm, and policy/reference/rollout placement to `train.jsonl`. The record must state that keeping the policy unsharded adds a full bf16 policy all-gather output per GPU and that beta-positive reference storage adds one bf16 shard per GPU.

- [ ] **Step 5: Run GRPO step and CLI tests**

Run: `pytest tests/test_grpo_step.py tests/test_cli_contracts.py -v`

Expected: PASS for beta-zero and beta-positive paths, global denominator, checkpoint boundary, phase peaks, and final recipe defaults.

- [ ] **Step 6: Commit GRPO**

```bash
git add train/fsdp_grpo.py tests/test_grpo_step.py tests/test_cli_contracts.py
git commit -m "feat: add full-parameter FSDP2 GRPO"
```

---

### Task 10: One-GPU SFT and Existing-GRPO Correctness Gates

**Files:**
- Create: `bench/correctness.py`
- Create: `tests/test_correctness_gate.py`
- Modify: `scripts/train_grpo_gsm8k.py:55-174`

**Interfaces:**
- Consumes: `train.fsdp_sft`, `train.fsdp_grpo`, the existing TRL GRPO script, fixed GSM8K subsets, three seeds, and consolidated small-model checkpoints.
- Produces: `run_sft_gate()`, `run_grpo_gate()`, `validate_gate_record()`, deterministic JSON records, and a nonzero exit status when a gate fails.

- [ ] **Step 1: Write failing gate-schema and acceptance tests**

```python
def test_gate_rejects_missing_seed_or_failed_metric():
    record = passing_gate_record()
    record["seeds"] = [41, 42]
    with pytest.raises(ValueError, match="exactly three seeds"):
        validate_gate_record(record)
    record = passing_gate_record()
    record["metrics"]["update_cosine_min"] = 0.998
    with pytest.raises(ValueError, match="0.999"):
        validate_gate_record(record)
```

- [ ] **Step 2: Run the test and verify the harness is missing**

Run: `pytest tests/test_correctness_gate.py -v`

Expected: collection error for `bench.correctness`.

- [ ] **Step 3: Add deterministic oracle controls without changing defaults**

Add `--seed` and `--run_record` to `scripts/train_grpo_gsm8k.py`, pass `seed` into `GRPOConfig`, and write the resolved config/log history only when `--run_record` is supplied. Existing invocations without the new flags must produce the same configuration as before.

- [ ] **Step 4: Implement the SFT gate**

Use `Qwen/Qwen2.5-0.5B-Instruct`, the first fixed 16 GSM8K rows, 20 optimizer steps, seeds 41/42/43, identical bf16 dtype, batches, AdamW, scheduler, attention backend, activation-checkpoint setting, and deterministic flags. Compare `--sharding none` against FSDP2 world size 1 for first-batch loss, every pre-clip norm, selected tensor deltas, and full loss curves. Require first-loss absolute error at most `5e-3`, per-step gradient-norm relative error at most `1e-2`, minimum selected-update cosine above `0.999`, and each mean loss-curve difference at most `2 * sqrt(var_unsharded / 3 + var_fsdp2 / 3) + 1e-3`. Write `results/fsdp_correctness/sft_gate.json` only if all three seeds complete.

- [ ] **Step 5: Implement the GRPO gate**

Run the fixed-rollout objective test, then compare `scripts/train_grpo_gsm8k.py` with `train.fsdp_grpo` at world size 1 using the same small model, subset, eight generations, reward, Dr. GRPO normalization, beta zero, temperature one, and seeds 41/42/43. Load selected FSDP update tensors from DCP with `get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` before comparing them, so this gate does not depend on the later standalone consolidator. Require fixed-rollout loss absolute error at most `5e-4`, gradient-norm relative error at most `1e-2`, minimum selected-update cosine above `0.999`, and each loss/reward mean-curve difference at most `2 * sqrt(var_oracle / 3 + var_fsdp2 / 3) + 1e-3`. Require resume next-step loss absolute error at most `1e-6`. Write `results/fsdp_correctness/grpo_gate.json`.

- [ ] **Step 6: Run fast schema tests**

Run: `pytest tests/test_correctness_gate.py -v`

Expected: PASS for missing seeds, mismatched model/dataset digests, failed tolerances, failed resume, and both passing record shapes.

- [ ] **Step 7: Commit the gate harness, not generated measurements**

```bash
git add bench/correctness.py tests/test_correctness_gate.py scripts/train_grpo_gsm8k.py
git commit -m "test: add FSDP2 correctness gates"
```

---

### Task 11: DCP-to-Hugging-Face Consolidation

**Files:**
- Create: `scripts/consolidate_dcp.py`
- Create: `tests/test_consolidate_dcp.py`
- Create: `tests/workers/consolidate_worker.py`

**Interfaces:**
- Consumes: a successful model DCP directory, source model/tokenizer metadata, output directory, and torchrun context.
- Produces: one Hugging Face model directory containing config/tokenizer assets, sharded safetensors plus index, `consolidation_manifest.json`, and verification results.

- [ ] **Step 1: Write a failing round-trip test**

```python
@pytest.mark.cuda
@pytest.mark.distributed
def test_consolidated_directory_loads_with_transformers(torchrun_result, tmp_path):
    row = torchrun_result("tests/workers/consolidate_worker.py", nproc=2, output_dir=tmp_path)
    assert row["optimizer_loaded"] is False
    assert row["hf_reload_ok"] is True
    assert row["parameter_count_match"] is True
    assert row["selected_hashes_match"] is True
    assert row["fixed_logits_max_abs_error"] < 1e-5
```

- [ ] **Step 2: Run the test and verify the script is missing**

Run: `pytest tests/test_consolidate_dcp.py -v`

Expected: FAIL because `scripts/consolidate_dcp.py` does not exist.

- [ ] **Step 3: Implement model-only rank-zero CPU gathering**

Recreate and shard the model exactly as training did, load only the DCP `model` state, then call:

```python
options = StateDictOptions(full_state_dict=True, cpu_offload=True)
full_state = get_model_state_dict(model, options=options)
```

PyTorch 2.8 returns the CPU full state only on rank 0 when both options are true; other ranks receive an empty mapping. Never request optimizer state. Before gathering, preflight host RAM for at least model bf16 bytes plus 25% workspace.

- [ ] **Step 4: Write the normal Hugging Face directory and verify it**

On rank 0, call `save_pretrained(output_dir, state_dict=full_state, safe_serialization=True, max_shard_size="5GB")`, save tokenizer/chat template/generation config, reload with `AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, device_map="cpu")`, compare parameter count and selected SHA-256 tensor hashes, and compare fixed-input logits with the distributed DCP model. Write `_SUCCESS` only after all checks pass.

- [ ] **Step 5: Run the tiny-model consolidation test**

Run: `pytest tests/test_consolidate_dcp.py -v`

Expected: PASS and a directory containing `config.json`, tokenizer files, `model.safetensors` or a safetensor index, `consolidation_manifest.json`, and `_SUCCESS`.

- [ ] **Step 6: Commit the consolidator**

```bash
git add scripts/consolidate_dcp.py tests/test_consolidate_dcp.py tests/workers/consolidate_worker.py
git commit -m "feat: consolidate DCP into Hugging Face checkpoints"
```

---

### Task 12: Parameterized Single-Node Slurm/torchrun Launcher

**Files:**
- Create: `scripts/launch_slurm.sh`
- Create: `tests/test_launch_slurm.py`

**Interfaces:**
- Consumes: `--stage`, `--gpus`, optional `--dry-run`, Slurm environment, and trailing stage arguments.
- Produces: validated torchrun commands for SFT, GRPO, correctness, consolidation, and individual scaling workers; scaling-controller mode uses the allocated eight GPUs sequentially.

- [ ] **Step 1: Write failing dry-run tests**

```python
@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
def test_launcher_builds_one_node_torchrun(gpus):
    result = subprocess.run(
        ["bash", "scripts/launch_slurm.sh", "--stage", "sft", "--gpus", str(gpus), "--dry-run", "--", "--output_dir", "/tmp/sft"],
        check=True, capture_output=True, text=True,
        env={**os.environ, "SLURM_NNODES": "1", "SLURM_JOB_ID": "dry"},
    )
    assert f"--nproc-per-node={gpus}" in result.stdout
    assert "--module train.fsdp_sft" in result.stdout
```

- [ ] **Step 2: Run the tests and verify the launcher is missing**

Run: `pytest tests/test_launch_slurm.py -v`

Expected: FAIL because `scripts/launch_slurm.sh` does not exist.

- [ ] **Step 3: Implement argument validation and module selection**

Map `sft -> train.fsdp_sft`, `grpo -> train.fsdp_grpo`, `correctness -> bench.correctness`, `consolidate -> scripts.consolidate_dcp`, and `scaling-worker -> bench.scaling`. Map `scaling` to the `bench.scaling` controller and require `--gpus 8` for its complete 1/2/4/8 sweep. Reject GPU counts outside 1/2/4/8, `SLURM_NNODES` other than one, mismatch between visible devices and `--gpus`, and unknown stages. Source `scripts/activate_great_lakes.sh`, run `scripts/cluster_check.py`, print all relevant environment/config values, and launch workers with:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node="$GPUS" --module "$MODULE" "${STAGE_ARGS[@]}"
```

Launch controller mode with `python -m bench.scaling "${STAGE_ARGS[@]}"`; that controller creates the four isolated torchrun subprocesses.

Set `NCCL_DEBUG=INFO`, `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`, `TORCH_NCCL_BLOCKING_WAIT=1`, and a configurable distributed timeout. Preserve shell arrays so values with spaces are not re-parsed.

- [ ] **Step 4: Add analytical and detected-VRAM preflight**

For SFT/GRPO, invoke the memory-model preflight before the model is constructed. Require A100-80 for the native-bf16 world-size-1 Qwen3 point; reject the fp32-resident profile at world size one; print warnings for the two-GPU 40 GiB/A40 margins specified by the design.

- [ ] **Step 5: Run launcher tests and shell syntax check**

Run: `bash -n scripts/launch_slurm.sh`

Run: `pytest tests/test_launch_slurm.py -v`

Expected: PASS for 1/2/4/8, invalid count, non-single-node allocation, stage mapping, trailing args, dry run without modules, and NCCL environment output.

- [ ] **Step 6: Commit the launcher**

```bash
git add scripts/launch_slurm.sh tests/test_launch_slurm.py
git commit -m "feat: add single-node FSDP2 Slurm launcher"
```

---

### Task 13: Fixed-Global-Batch Scaling and Communication Accounting

**Files:**
- Create: `bench/scaling.py`
- Modify: `train/memory_model.py`
- Create: `tests/test_scaling_metrics.py`
- Modify: `tests/test_cli_contracts.py`

**Interfaces:**
- Consumes: the real `sft_optimizer_step()`, world sizes 1/2/4/8, fixed packed dataset, GPU hardware identity, and profiler traces.
- Produces: `ScalingConfig`, `run_worker()`, `run_sweep_controller()`, `deduplicated_storage_bytes()`, `measure_memory_components()`, `merge_intervals()`, `communication_fractions()`, `compute_mfu()`, `scaling_efficiency()`, `scaling.jsonl`, `scaling.csv`, rank JSONL, and per-run configs.

- [ ] **Step 1: Write failing metric tests with synthetic profiler intervals**

```python
def test_exposed_communication_subtracts_compute_overlap():
    comm = [(0, 10), (20, 30)]
    compute = [(5, 25)]
    active, exposed = communication_fractions(comm, compute, step_window=(0, 40))
    assert active == pytest.approx(0.50)
    assert exposed == pytest.approx(0.25)


def test_mfu_uses_aggregate_dense_peak():
    mfu = compute_mfu(parameters=8_190_735_360, useful_tokens=16_384, seconds=10.0, world_size=4, peak_bf16_tflops_per_gpu=312.0)
    assert mfu == pytest.approx(6 * 8_190_735_360 * 16_384 / (10 * 4 * 312e12))


def test_storage_bytes_deduplicates_views():
    tensor = torch.zeros(32, dtype=torch.bfloat16)
    assert deduplicated_storage_bytes([tensor, tensor.view(4, 8)]) == 64
```

- [ ] **Step 2: Run the tests and verify the benchmark is missing**

Run: `pytest tests/test_scaling_metrics.py -v`

Expected: collection error for `bench.scaling`.

- [ ] **Step 3: Implement worker measurement using the real SFT step**

Use global batch eight, sequence length 2048, local microbatch one, and accumulation `{1: 8, 2: 4, 4: 2, 8: 1}`. Use the same Qwen revision, packed subset digest, seed, checkpoint flag, optimizer, and dtype for every point. Initialize AdamW state, run three warmup steps, reset allocated/reserved peaks, run ten unprofiled measured steps, synchronize around the measurement window, and aggregate non-padding tokens and per-rank allocated/reserved peaks.

For the memory ledger, inventory deduplicated local storage bytes for DTensor parameter shards, gradient shards after backward, and AdamW tensor states after initialization. Wrap one separate probe step with `torch.autograd.graph.saved_tensors_hooks` to track the maximum simultaneously live saved-tensor bytes as measured activations. Register observation hooks after FSDP's hooks on each explicit group; the pre-forward observation sees the unsharded/prefetched group allocation and the post-forward observation sees the resharded baseline. Record the maximum allocator delta as measured collective/all-gather workspace. Define measured `other` as total allocated peak minus the four measured categories, clamped at zero, and retain the raw counters so this residual can be audited.

- [ ] **Step 4: Implement separate profiler iterations and interval unions**

Run three additional steps under `torch.profiler` and classify CUDA events whose names identify NCCL as communication. Treat remaining non-memcpy/non-memset CUDA kernels as compute. Merge overlapping intervals before calculating communication-active duration and communication duration not overlapped by compute. Keep profiler results out of throughput timing.

- [ ] **Step 5: Implement hardware-aware MFU and sweep control**

Use dense bf16 peaks of 312 TFLOP/s for A100 and 149.7 TFLOP/s for A40, with `--peak_bf16_tflops` required for unknown names. The controller must require one homogeneous eight-GPU allocation, run 1/2/4/8 torchrun workers sequentially using the first N visible GPUs, verify identical GPU names and run config digests, then compute `throughput_N / (N * throughput_1)`. Do not merge heterogeneous device results.

- [ ] **Step 6: Write records in the repository's append-only style**

Each record must include run ID, code commit/dirty state, Slurm job ID, GPU name/VRAM/topology, world size, CUDA/NCCL/package versions, model revision, dataset digest, seed, batch shape, checkpoint flag, per-rank and max memory, measured params/grads/optimizer/activations/collective/other bytes with probe method, useful tokens/sec, MFU definition/value, communication definitions/values, step mean/dispersion, and scaling efficiency. Rank 0 writes `scaling.jsonl`, stable-field `scaling.csv`, `run_config.json`, and rank JSONL.

- [ ] **Step 7: Run pure tests and a tiny two-GPU benchmark smoke**

Run: `pytest tests/test_scaling_metrics.py tests/test_cli_contracts.py -v`

Run: `python -m tests.tiny_qwen --output_dir /tmp/tiny-qwen3`

Run on two GPUs: `torchrun --standalone --nproc-per-node=2 --module bench.scaling --worker --model /tmp/tiny-qwen3 --sequence_length 64 --global_batch_size 4 --warmup_steps 1 --measure_steps 2 --profile_steps 1 --output_dir /tmp/fsdp-scaling-smoke`

Expected: PASS; the smoke record must contain both ranks, nonzero tokens/sec, allocated/reserved peaks, MFU, and both communication fractions.

- [ ] **Step 8: Commit the benchmark**

```bash
git add bench/scaling.py train/memory_model.py tests/test_scaling_metrics.py tests/test_cli_contracts.py
git commit -m "feat: add FSDP2 scaling benchmark"
```

---

### Task 14: Execute and Commit the Correctness Gates Before Scaling

**Files:**
- Create from real runs: `results/fsdp_correctness/sft_gate.json`
- Create from real runs: `results/fsdp_correctness/grpo_gate.json`

**Interfaces:**
- Consumes: a one-GPU A40 or A100 Slurm allocation, committed Tasks 1-13 code, cached Qwen2.5-0.5B, and GSM8K.
- Produces: two passing committed gate records that `bench.scaling` requires before accepting Qwen3-8B.

- [ ] **Step 1: Run the SFT gate on one GPU**

```bash
srun bash scripts/launch_slurm.sh --stage correctness --gpus 1 -- \
  --gate sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --seeds 41,42,43 \
  --dataset_limit 16 \
  --max_steps 20 \
  --output_dir results/fsdp_correctness
```

Expected: exit 0 and `sft_gate.json` with `passed=true`.

- [ ] **Step 2: Run the GRPO gate on one GPU**

```bash
srun bash scripts/launch_slurm.sh --stage correctness --gpus 1 -- \
  --gate grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --seeds 41,42,43 \
  --dataset_limit 16 \
  --max_steps 20 \
  --num_generations 8 \
  --output_dir results/fsdp_correctness
```

Expected: exit 0 and `grpo_gate.json` with `passed=true`.

- [ ] **Step 3: Validate records and refuse scaling on failure**

Run: `python -m bench.correctness --validate results/fsdp_correctness/sft_gate.json results/fsdp_correctness/grpo_gate.json`

Expected: PASS and a printed summary of three seeds, loss tolerances, minimum update cosine, pooled curve bands, and resume check. If validation exits nonzero, stop here; do not run Task 15.

- [ ] **Step 4: Commit the real gate evidence**

```bash
git add results/fsdp_correctness/sft_gate.json results/fsdp_correctness/grpo_gate.json
git commit -m "results: record passing FSDP2 correctness gates"
```

---

### Task 15: Run the Homogeneous Qwen3 Scaling and Checkpointing Ablation

**Files:**
- Create from real runs: `results/fsdp_scaling/scaling.jsonl`
- Create from real runs: `results/fsdp_scaling/scaling.csv`
- Create from real runs: `results/fsdp_scaling/ranks.jsonl`
- Create from real runs: `results/fsdp_scaling/run_config.json`
- Create from real runs: `results/fsdp_scaling/activation_checkpointing.jsonl`

**Interfaces:**
- Consumes: committed passing gate records and one homogeneous 8xA100-80 single-node Slurm allocation.
- Produces: measured 1/2/4/8 scaling records and matched world-size-4 activation-checkpointing on/off records.

- [ ] **Step 1: Verify hardware homogeneity and gate commits**

Run: `nvidia-smi --query-gpu=name,memory.total,pci.bus_id --format=csv,noheader`

Run: `python -m bench.correctness --validate results/fsdp_correctness/sft_gate.json results/fsdp_correctness/grpo_gate.json`

Expected: eight identical A100 80 GiB names/capacities and both gates passing from committed files.

- [ ] **Step 2: Run the 1/2/4/8 controller**

```bash
bash scripts/launch_slurm.sh --stage scaling --gpus 8 -- \
  --world_sizes 1,2,4,8 \
  --model Qwen/Qwen3-8B \
  --global_batch_size 8 \
  --sequence_length 2048 \
  --micro_batch_size 1 \
  --warmup_steps 3 \
  --measure_steps 10 \
  --profile_steps 3 \
  --activation_checkpointing \
  --output_dir results/fsdp_scaling
```

Expected: four successful records with identical non-world-size config digests and accumulation 8/4/2/1.

- [ ] **Step 3: Run the matched world-size-4 checkpointing ablation**

```bash
torchrun --standalone --nproc-per-node=4 --module bench.scaling --worker \
  --model Qwen/Qwen3-8B \
  --global_batch_size 8 \
  --sequence_length 2048 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 3 \
  --measure_steps 10 \
  --profile_steps 3 \
  --no_activation_checkpointing \
  --record_stem activation_checkpointing \
  --output_dir results/fsdp_scaling
```

The matching checkpointing-on world-size-4 row comes from the main sweep. If the off run OOMs, record the allocator OOM, last successful phase, and observed peak as an OOM measurement instead of inventing throughput.

- [ ] **Step 4: Validate records before documentation**

Run: `python -m bench.scaling --validate results/fsdp_scaling`

Expected: PASS for four world sizes, fixed global batch, homogeneous hardware, complete rank peaks, profiler definitions, and matched ablation settings.

- [ ] **Step 5: Commit the real scaling evidence**

```bash
git add results/fsdp_scaling/scaling.jsonl results/fsdp_scaling/scaling.csv results/fsdp_scaling/ranks.jsonl results/fsdp_scaling/run_config.json results/fsdp_scaling/activation_checkpointing.jsonl
git commit -m "results: record Qwen3 FSDP2 scaling sweep"
```

---

### Task 16: Generate Provenance-Checked Memory and Scaling Documents

**Files:**
- Create: `bench/reporting.py`
- Create: `tests/test_report_provenance.py`
- Create: `docs/memory_ledger.md`
- Create: `docs/fsdp_notes.md`

**Interfaces:**
- Consumes: committed correctness, scaling, rank, and checkpoint-ablation records plus analytical functions from `train.memory_model`.
- Produces: deterministic Markdown, `validate_document_provenance()`, and a nonzero exit when a measured statement lacks a committed run record.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_measured_claim_requires_existing_run_id(tmp_path):
    document = tmp_path / "notes.md"
    document.write_text("Measured peak was 24.1 GiB [run:missing].\n")
    with pytest.raises(ValueError, match="missing"):
        validate_document_provenance(document, known_run_ids={"present"})


def test_prediction_must_precede_measurement_in_memory_rows():
    row = render_memory_row(predicted=22.96, measured_allocated=24.1, measured_reserved=25.0, run_id="sft-w4")
    assert row.index("22.96") < row.index("24.1")
```

- [ ] **Step 2: Run the tests and verify reporting is missing**

Run: `pytest tests/test_report_provenance.py -v`

Expected: collection error for `bench.reporting`.

- [ ] **Step 3: Implement deterministic report rendering**

`docs/memory_ledger.md` must present assumptions and formulas first, then one row per world size where each component is written as `predicted / measured`: params, grads, optimizer, activations, all-gather/gradient buffers, other, total allocated, and total reserved, followed by delta and `[run:<id>]`. Identify direct storage inventories, saved-tensor live-byte tracking, hook-observed collective deltas, and allocator residual as distinct measurement methods. Explain deltas using recorded allocator reservations, logits/loss workspace, collective overlap, fragmentation, and profiler-observed behavior only where supported by that record.

`docs/fsdp_notes.md` must show tokens/sec, MFU, communication-active fraction, exposed communication fraction, scaling efficiency, and the point where efficiency degrades. Tie the mechanism to the measured communication/compute fractions. Include the checkpointing on/off memory delta and throughput delta at world size four; if off OOMed, state the measured OOM and omit a throughput-loss number.

- [ ] **Step 4: Generate documents from committed records**

Run: `python -m bench.reporting --correctness_dir results/fsdp_correctness --scaling_dir results/fsdp_scaling --output_docs docs`

Expected: both required Markdown files are created and every measured paragraph or table row contains a run ID.

- [ ] **Step 5: Validate links, record existence, and no unsupported empirical language**

Run: `pytest tests/test_report_provenance.py -v`

Run: `python -m bench.reporting --validate docs/memory_ledger.md docs/fsdp_notes.md --record_dirs results/fsdp_correctness results/fsdp_scaling`

Expected: PASS with every cited run ID present in committed JSON and no prediction labeled as measured.

- [ ] **Step 6: Commit renderer and generated documents**

```bash
git add bench/reporting.py tests/test_report_provenance.py docs/memory_ledger.md docs/fsdp_notes.md
git commit -m "docs: record FSDP2 memory and scaling results"
```

---

### Task 17: End-to-End SFT → GRPO → Consolidation → Existing Deployment Smoke

**Files:**
- No changes to existing quantization or serving code.
- Create from real run: output DCP/Hugging Face directories outside Git.
- Modify only if needed for instructions: `README.md`

**Interfaces:**
- Consumes: the implemented FSDP2 training track, passing gates, a feasible Slurm allocation, and existing `scripts/quantize_awq.py` and `scripts/serve.py` interfaces.
- Produces: a Qwen3-8B SFT DCP, GRPO DCP, consolidated Hugging Face directory, AWQ directory, and successful vLLM offline completion.

- [ ] **Step 1: Run a bounded Qwen3 SFT artifact smoke**

```bash
srun bash scripts/launch_slurm.sh --stage sft --gpus 8 -- \
  --model Qwen/Qwen3-8B \
  --dataset_limit 64 \
  --sequence_length 2048 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_steps 2 \
  --save_steps 2 \
  --output_dir "$PQS_ROOT/ckpts/qwen3_8b_fsdp_sft_smoke"
```

Expected: successful step-2 DCP with `_SUCCESS` and a resume run that reproduces the uninterrupted next-step loss.

- [ ] **Step 2: Run a bounded beta-zero GRPO artifact smoke from SFT**

```bash
srun bash scripts/launch_slurm.sh --stage grpo --gpus 8 -- \
  --model "$PQS_ROOT/ckpts/qwen3_8b_fsdp_sft_smoke/step-00000002" \
  --dataset_limit 8 \
  --num_generations 8 \
  --max_completion_length 128 \
  --max_steps 1 \
  --save_steps 1 \
  --beta 0.0 \
  --output_dir "$PQS_ROOT/ckpts/qwen3_8b_fsdp_grpo_smoke"
```

Expected: successful step-1 DCP whose run record says policy `fsdp2_sharded_gpu`, reference `absent_beta_zero`, and completed rollout tensors `cpu_after_group`.

- [ ] **Step 3: Consolidate the GRPO policy**

```bash
srun bash scripts/launch_slurm.sh --stage consolidate --gpus 8 -- \
  --checkpoint "$PQS_ROOT/ckpts/qwen3_8b_fsdp_grpo_smoke/step-00000001" \
  --output_dir "$PQS_ROOT/ckpts/qwen3_8b_fsdp_grpo_smoke_hf"
```

Expected: a `_SUCCESS` Hugging Face directory that reloads, matches hashes/logits, and contains no optimizer state.

- [ ] **Step 4: Feed the directory into unchanged downstream scripts**

```bash
python scripts/quantize_awq.py \
  --model_name_or_path "$PQS_ROOT/ckpts/qwen3_8b_fsdp_grpo_smoke_hf" \
  --output_dir "$PQS_ROOT/ckpts_awq/qwen3_8b_fsdp_grpo_smoke_w4g128" \
  --calib_limit 16 \
  --max_calib_seq_len 512

python scripts/serve.py \
  --mode offline \
  --model "$PQS_ROOT/ckpts_awq/qwen3_8b_fsdp_grpo_smoke_w4g128" \
  --quantization awq \
  --max-new-tokens 64
```

Expected: AWQ completes using the existing `--model_name_or_path` interface and vLLM produces a completion using the existing `--model` interface, with no edits to either script.

- [ ] **Step 5: Run the complete verification suite**

Run: `pytest -m "not slurm" -v`

Run: `python -m bench.correctness --validate results/fsdp_correctness/sft_gate.json results/fsdp_correctness/grpo_gate.json`

Run: `python -m bench.scaling --validate results/fsdp_scaling`

Run: `python -m bench.reporting --validate docs/memory_ledger.md docs/fsdp_notes.md --record_dirs results/fsdp_correctness results/fsdp_scaling`

Expected: all tests and artifact validators pass. Confirm `rg -n "FullyShardedDataParallel|accelerate launch" train scripts/launch_slurm.sh scripts/consolidate_dcp.py` returns no implementation-path matches; the correctness harness may invoke the existing Accelerate-based GRPO oracle but never wraps the new FSDP2 trainers with it.

- [ ] **Step 6: Document the entry points without altering deployment behavior**

Add a compact README section with SFT, GRPO, resume, consolidation, scaling, and existing quantize/serve commands. State the hardware feasibility table and link the two provenance-backed documents.

- [ ] **Step 7: Commit final integration instructions**

```bash
git add README.md
git commit -m "docs: add FSDP2 end-to-end workflow"
```

---

## Execution Review Gates

After Tasks 1-5, review the shared interfaces and two-GPU tiny-model DCP round trip before building either trainer.

After Tasks 6-11, review mathematical parity, beta-zero/reference allocation, and consolidation before adding launch or benchmark code.

After Task 14, stop unless both committed correctness records pass. Task 15 is forbidden before that gate.

After Task 15, inspect raw JSON/JSONL records before generating either documentation file. Generated documents are not hand-edited to create empirical claims.

After Task 17, use the verification-before-completion workflow and report the exact test commands, Slurm job IDs, run IDs, checkpoint paths, and commit hashes.

## Version-Sensitive References

- PyTorch 2.8 FSDP2 `fully_shard`: https://docs.pytorch.org/docs/2.8/distributed.fsdp.fully_shard.html
- PyTorch 2.8 Distributed Checkpoint: https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html
- Qwen3 Transformers model contract: https://huggingface.co/docs/transformers/model_doc/qwen3
- Qwen3-8B immutable configuration fields: https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- TRL GRPO trainer semantics: https://huggingface.co/docs/trl/grpo_trainer
