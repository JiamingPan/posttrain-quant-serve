# Day 4 — Quantization Pipeline, AWQ, and 1.5B Handoff

Day 4 closed the 0.5B quantization smoke loop and started the real 1.5B path.
The main conclusion is now clear: Qwen2.5-0.5B is useful as a cheap dev loop, but
it is too close to the W4 quantization floor to support the final research claim.
The final result should be reported on Qwen2.5-1.5B-Instruct.

## What Finished

- Built and ran the bnb-NF4 W4 eval path on the 0.5B base and `g8_dr100` GRPO
  checkpoint.
- Built the AWQ W4G128 quantization path with AutoAWQ.
- Quantized both 0.5B checkpoints with AWQ:
  - base: `qwen2_5_0_5b_base_awq_w4g128`
  - GRPO: `qwen2_5_0_5b_g8_dr100_awq_w4g128`
- Fixed AWQ eval loading:
  - added `gptqmodel` to `requirements-awq.txt`
  - forced `dtype=torch.float16` when loading AWQ checkpoints, because the Marlin
    AWQ kernels reject bf16 activations
- Completed the 0.5B FP16-vs-AWQ eval.
- Completed the 1.5B GRPO reproduction run:
  - job: `51163281`
  - state: `COMPLETED 0:0`
  - output: `ckpts/qwen2_5_1_5b_grpo_g8_dr100`
- Completed the 1.5B base AWQ job:
  - job: `51163933`
  - state: `COMPLETED 0:0`
  - output: `ckpts_awq/qwen2_5_1_5b_base_awq_w4g128`
- Completed the 1.5B GRPO AWQ job:
  - job: `51164123`
  - state: `COMPLETED 0:0`
  - output: `ckpts_awq/qwen2_5_1_5b_g8_dr100_awq_w4g128`
- Completed the 1.5B FP16/bnb-W4/AWQ eval matrix:
  - job: `51173013`
  - state: `COMPLETED 0:0`
  - output: `evals/gsm8k_compare_test100_qwen2_5_1_5b_g8_dr100_fp16_w4_awq`
- Completed weight/outlier diagnostics for 0.5B and 1.5B base-vs-GRPO.

## 0.5B Results

### bnb-NF4 W4

| Variant | FP16 Acc | bnb-W4 Acc |
| --- | ---: | ---: |
| Base 0.5B | 0.06 | 0.12 |
| GRPO 0.5B | 0.22 | 0.10 |

Key metrics:

- `delta_fp16 = +0.16`
- `delta_w4 = -0.02`
- `gain_survival_w4 = -0.18`

### AWQ W4G128

| Variant | FP16 Acc | AWQ Acc |
| --- | ---: | ---: |
| Base 0.5B | 0.06 | 0.06 |
| GRPO 0.5B | 0.22 | 0.02 |

Key metrics:

- `delta_fp16 = +0.16`
- `delta_awq = -0.04`
- `gain_survival_awq = -0.20`
- `quant_drop_base_awq = 0.00`
- `quant_drop_g8_awq = +0.20`

Interpretation: both bnb-NF4 and AWQ erase the observed 0.5B GRPO FP16 gain.
This is not the final answer to the research question. It is a quantization-floor
diagnostic: 0.5B is too small or too fragile for W4 to preserve meaningful signal.
Keep this as a documented negative/dev-loop result.

### 0.5B Weight Diagnostics

The 0.5B weight diagnostic did **not** show a meaningful base-vs-GRPO shift in
the raw weight distribution:

| Metric | Base | GRPO `g8_dr100` | Interpretation |
| --- | ---: | ---: | --- |
| channel outlier frac | 0.000221815 | 0.000221815 | identical |
| abs outlier frac | 0.002047758 | 0.002059751 | tiny increase |
| W4 relative MSE | 0.016417633 | 0.016417381 | tiny decrease |
| W4 SNR | 17.5942866 dB | 17.5943422 dB | tiny increase |

So the 0.5B W4 failure is not explained by an obvious global weight-outlier
explosion after GRPO. It looks more like model/eval fragility near the W4 floor.

## Why 1.5B Is Now the Main Path

The project question is whether RL post-training changes how a model quantizes.
On 0.5B, both base and GRPO are near the accuracy floor after W4, so "GRPO did not
survive quantization" is confounded with "0.5B is not quantizable enough at W4."

Qwen2.5-1.5B-Instruct still fits on one A40, but should have enough redundancy
that base W4/AWQ remains functional. That makes the base-vs-GRPO W4/AWQ comparison
meaningful.

## 1.5B Result

The 1.5B matrix is now the main result candidate:

| Variant | Accuracy | Meaning |
| --- | ---: | --- |
| `base_fp16` | 0.06 | original Qwen2.5-1.5B-Instruct |
| `g8_dr100_fp16` | 0.19 | 1.5B GRPO checkpoint |
| `base_w4` | 0.17 | base with bnb-NF4 W4 |
| `g8_dr100_w4` | 0.31 | GRPO with bnb-NF4 W4 |
| `base_awq` | 0.23 | base with AWQ W4G128 |
| `g8_dr100_awq` | 0.35 | GRPO with AWQ W4G128 |

Headline deltas:

- `delta_fp16 = +0.13`
- `delta_w4 = +0.14`
- `delta_awq = +0.12`
- `gain_survival_w4 = +0.01`
- `gain_survival_awq = -0.01`

Interpretation: GRPO changed model behavior enough to improve GSM8K accuracy,
and that gain mostly survived both bnb-W4 and AWQ. The quantized models scoring
higher than FP16 is surprising, so this should be treated as a result to audit
with paired examples and a larger eval slice, not as a claim that quantization
generally improves the model.

## 1.5B Weight Diagnostics

The 1.5B weight diagnostic also did **not** show a meaningful base-vs-GRPO shift:

| Metric | Base | GRPO `g8_dr100` | Interpretation |
| --- | ---: | ---: | --- |
| max abs max | 3.062500 | 3.062500 | basically identical |
| channel outlier frac | 0.000164523 | 0.000170327 | tiny increase |
| abs outlier frac | 0.001281860 | 0.001289095 | tiny increase |
| W4 relative MSE | 0.015504525 | 0.015504335 | tiny decrease |
| W4 SNR | 17.9587079 dB | 17.9587659 dB | tiny increase |

Current conclusion:

> GRPO changed model behavior enough to improve GSM8K accuracy, and that gain
> mostly survived W4/AWQ. The weight diagnostics do not show a big global
> outlier/distribution shift after GRPO, so there is no evidence yet that this
> GRPO recipe makes the weights harder to quantize.

## Next Step

Do not spend more compute on 0.5B except for documentation or notebook cleanup.
The next useful work is to audit the 1.5B matrix and, if time permits, run a
larger eval slice to check whether the surprising quantized-over-FP16 behavior
persists.

```bash
MODEL_TAG=qwen2_5_1_5b \
PRECISIONS=fp16,w4,awq \
EVAL_SPLIT=test \
EVAL_LIMIT=200 \
EVAL_OUTPUT_DIR=/scratch/huterer_root/huterer0/jiamingp/pqs/evals/gsm8k_compare_test200_qwen2_5_1_5b_g8_dr100_fp16_w4_awq \
sbatch --job-name=pqs-eval-1p5-test200 \
  --account=cavestru0 \
  --time=02:00:00 \
  --export=ALL \
  slurm/eval_gsm8k_compare.sbatch
```

The headline numbers for the final study are:

- `delta_fp16`
- `delta_w4`
- `delta_awq`
- `gain_survival_w4`
- `gain_survival_awq`

The key question is whether `gain_survival_awq` stays near zero or positive on
1.5B. If it does, GRPO gain survives calibration-aware W4 quantization. If it is
strongly negative again while base-AWQ is still functional, then GRPO likely made
the checkpoint harder to quantize.

---

## The eval was measuring through a stopping bug (found after the runs above)

The 1.5B matrix above is **not yet trustworthy**, and a later audit run
(`..._max256_audit`) made the reason concrete. Every row of the audit matrix had:

- `hit_max_new_tokens_rate = 1.0` — every generation was cut off at the token cap,
- `ended_with_eos_rate = 0.0` — no generation ever stopped on its own.

That is the fingerprint of a prompt-format bug, not a model or quantization effect.

### What the bug is

Qwen2.5-Instruct is a **chat** model. It only emits its end-of-turn stop token
`<|im_end|>` when the input is wrapped in its chat template
(`<|im_start|>user … <|im_start|>assistant`). Our prompts were **raw text**
("Solve the math problem… Problem: … Solution:") with no chat wrapping. With no
chat structure, the model never produced a stop token, so every completion ran to
`max_new_tokens`. (`hit_max_new_tokens_rate = 1.0` and `ended_with_eos_rate = 0.0`.)

**Two separate consequences of this one root cause — keep them distinct:**

1. **No `<|im_end|>` → never stops on its own.** True in *both* the early matrix and
   the audit matrix. By itself this wastes compute and makes the model ramble, but it
   does not directly zero out accuracy.
2. **Token cap too short → answer truncated before it is written.** This is what
   actually moved the number. A GSM8K solution with chain-of-thought needs space to
   reach the final `#### <answer>` line. The early matrix used a short cap (~96
   tokens), so the model was cut off *before it ever produced the answer* → the parser
   found nothing → accuracy floored at base_fp16 ≈ 0.06. The audit matrix used a larger
   cap (256), so the answer fit inside the (still un-stopped) ramble and got parsed →
   base_fp16 ≈ 0.52.

So the 0.06 → 0.52 swing is the **truncation-length** effect, not the stop-token
effect — both runs failed to stop, but only the short-cap run was cutting off the
answer. The fix needs *both*: emit `<|im_end|>` (so the model stops and we stop wasting
tokens / measure real stopping behavior) **and** enough budget (512) that a full
chain-of-thought reaches its answer line. Until both hold, the deltas do not mean
anything.

### The fix (Option B — train + eval consistency)

Standardize **both** `train_grpo_gsm8k.py` and `eval_gsm8k_compare.py` on
`tokenizer.apply_chat_template`, stop on `<|im_end|>` as well as `eos`, and give the
model room to finish (`max_new_tokens` 96 → 512). Success check before trusting any
number: `ended_with_eos_rate` should rise from 0.0 toward ~1.0 and
`hit_max_new_tokens_rate` should fall from 1.0 toward ~0. Full prompt:
`notes/codex-chat-template-standardize-prompt.md`. This retires the current
raw-trained `g8_dr100` checkpoints (they cannot be fairly compared against a
chat-formatted base) and requires a fresh 1.5B GRPO run.

---

## What the reward function is (and why the format change keeps coming up)

The **reward function** (`scripts/gsm8k_reward.py`) is the code that scores each
completion during GRPO. It is the entire learning signal: GRPO samples a group of
completions per problem, the reward function turns each into a number, and the
within-group spread of those numbers is what GRPO learns from.

For this project the reward function:

1. Extracts the final answer from the completion — looks for `#### <number>`, then
   falls back to `<answer>` tags, then "the answer is …", then the last number.
2. Returns ~1.0 if that number matches the GSM8K gold answer, 0 otherwise.
3. Subtracts a 0.25 penalty if the completion leaked prompt junk after the answer
   (rambled into a fake new "Problem:" / "Assistant:").

Because the reward is computed by *checking the math*, not by a learned reward model,
this is **RLVR (RL with Verifiable Rewards)** — the modern reasoning-model recipe and
the reason no separate reward model is needed.

This is why the chat-template change kept raising a flag: GRPO's whole signal flows
through the reward function. If the new chat format ever made the answer-extraction
fail, every completion would score 0, reward variance would go to zero, and we would
re-trigger the **advantage collapse** from Day 3 (std(r)=0 → all advantages=0 → no
learning → model degrades). A broken reward function does not crash; it silently stops
teaching. That is why the chat-template prompt gates on a tiny smoke run that prints
reward / reward-std / zero-variance-group fraction *before* spending GPU.

## Does the chat-template training "ruin what GRPO can study"? No — here is why

The worry: by changing the training prompt format (raw → chat), are we changing the
experiment so much that we are no longer studying the same thing? Three reasons the
answer is no, and the change actually *improves* the study's validity:

1. **The research question is format-agnostic.** The question is "does RL
   post-training change how a model quantizes?" That is about what GRPO does to the
   *weights* and whether that survives 4-bit — not about which prompt wrapper was
   used. The chat template changes how we *talk to* the model, not what RL is
   optimizing (still: maximize verifiable GSM8K reward via group-relative advantage).
   GRPO is the same algorithm with the same objective either way.

2. **Consistency removes a confound; it does not add one.** The risk to a clean study
   is *mismatch* — training in one format and evaluating in another, so a measured
   difference could be a format artifact instead of a real effect. Standardizing both
   sides on the chat template *eliminates* that confound. The earlier raw-prompt setup
   was the less valid one (chat model used out of distribution); the chat template is
   how the model was actually pretrained/instruct-tuned to operate, so it is the more
   faithful measurement.

3. **The reward and the GRPO mechanics are unchanged.** Same verifiable-reward grader,
   same `num_generations=8`, same `dr_grpo` loss, same `scale_rewards=none`, same
   `beta=0`. The only thing that moves is the prompt wrapper. The advantage-collapse
   safeguard (smoke-check reward variance first) ensures the format change did not
   quietly kill the learning signal. So what GRPO can study is preserved — we are just
   running the same study on a model that is now allowed to stop and is being used the
   way it was designed to be used.

One honest caveat to keep: because we retrain GRPO under the chat format, the new
checkpoint is a *different* trained model than the raw-format `g8_dr100`. We are not
re-scoring the old model in a new format (that would be unfair); we are running the
same recipe cleanly end-to-end. The old 0.5B/1.5B raw results stay in this log as the
documented stopping-bug / floor diagnostic that motivated the fix.

---

## Day 4 Closeout — What Is Actually Finished

Day 4 is finished. The main deliverable was not a final quantization claim; it was
the audit that made the pipeline trustworthy enough to produce one.

Finished:

- Found the root eval bug: raw text prompts were being fed to a chat model, so
  generations did not emit `<|im_end|>` and every row hit the token cap.
- Standardized train/eval/AWQ calibration on the Qwen chat template via the shared
  `build_gsm8k_chat_text` helper.
- Added stopping metrics to eval and verified the fix:
  - stop-check job: `51189211`
  - `base_fp16` test20 accuracy: `0.55`
  - `hit_max_new_tokens_rate = 0.0`
  - `ended_with_eos_rate = 1.0`
  - `prompt_leak_rate = 0.0`
- Verified chat-format GRPO reward still fires:
  - smoke job: `51189263`
  - `[reward_diag]` had nonzero reward std on multiple calls.
- Ran the old small-data chat GRPO recipe end-to-end:
  - train job: `51189288`
  - output: `ckpts/qwen2_5_1_5b_grpo_g8_dr100_chat`
  - test100 FP16: base `0.68`, GRPO `0.63`
  - train10 FP16: base `0.70`, GRPO `0.80`
  - interpretation: the 10-prompt recipe slightly improved the tiny training slice
    but did not generalize; this is overfit/recipe evidence, not a quantization
    conclusion.
- Made the final recipe decision: stay on GSM8K, do not switch to MATH for this
  ship, scale GRPO data from 10 to 1000 prompts, run once, and proceed to the
  quantization matrix regardless of the FP16 delta.
- Added `slurm/train_grpo.sbatch` for the final data1000 recipe:
  - commit: `5c5f9f2 Add GRPO data1000 training wrapper`
  - defaults: Qwen2.5-1.5B, `DATASET_LIMIT=1000`, `MAX_STEPS=250`,
    `NUM_GENERATIONS=8`, `GRAD_ACCUM=8`, `BETA=0.0`, `LOSS_TYPE=dr_grpo`,
    `SCALE_REWARDS=none`, `TEMPERATURE=1.0`, `MAX_COMPLETION_LENGTH=512`.
- Preflighted the new wrapper:
  - job: `51190636`
  - state: `COMPLETED 0:0`
  - `[reward_diag]` had nonzero reward std on calls 1 and 4.

The active work now moves to Day 5: finish the data1000 GRPO run, run held-out
FP16 eval, then run AWQ/bnb-W4 quantization comparison and write the honest result.
