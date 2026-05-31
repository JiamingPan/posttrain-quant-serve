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
