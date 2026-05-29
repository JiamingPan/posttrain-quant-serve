# GRPO / RLVR Literature Map

Annotated reading list for this project, organized by the problem it helps with.
Each entry says *why it matters for posttrain-quant-serve specifically*, not just
what the paper is. Read top-to-bottom; the first tier directly explains the Day 2
regression and the Day 3 fixes.

Project context for reference: Qwen2.5-0.5B-Instruct, TRL `GRPOTrainer`, GSM8K
verifiable reward, single A40. Day 2 result: 100-step checkpoint 0.60->0.50,
300-step leak-penalty checkpoint 0.60->0.20, reference PPL 1.878->1.934. The
research question is whether GRPO post-training changes quantization behavior.

---

## Tier 1 — Directly explains your regression and Day 3 fixes

### 1. DeepSeekMath / original GRPO (arXiv:2402.03300)
The paper that introduced GRPO. Read for the exact objective: group-relative
advantage (no critic), the per-token policy-gradient loss, and the KL term
against the frozen reference. You already have this PDF in the repo.
- Why it matters here: it is the algorithm `GRPOTrainer` implements. Knowing the
  objective is what lets you reason about *why* advantage can go to zero and how
  KL is supposed to anchor the policy.
- Extract: the advantage normalization (mean/std within group), the KL estimator,
  and the role of `num_generations` (group size) in advantage variance.
- https://arxiv.org/abs/2402.03300

### 2. Understanding R1-Zero-Like Training: A Critical Perspective — "Dr. GRPO" (arXiv:2503.20783, COLM 2025)
The single most relevant analysis paper for your situation. It identifies an
**optimization bias in GRPO that artificially inflates response length,
especially for *incorrect* outputs**, caused by the response-length and
std-normalization terms in the objective. Dr. GRPO removes those two terms to
recover the unbiased PPO objective.
- Why it matters here: your 300-step run got worse and your notebook tracks
  completion length / leakage. The length-inflation-on-wrong-answers bias is a
  prime suspect for "trained model rambles into wrong answers and regresses."
  Worth checking whether your TRL version applies length normalization, and
  whether the Dr. GRPO change is exposed as a config flag.
- Extract: which two normalization terms to drop, and the minimalist
  Qwen2.5-Math recipe (they used a small Qwen, like you).
- https://arxiv.org/abs/2503.20783

### 3. "A brief example of reward hacking in GRPO" (Mukherjee, blog)
A worked, small-scale case study using **HF `GRPOTrainer` on Qwen2.5-1.5B-Instruct**
— almost identical stack to yours. Shows that when the GRPO advantage becomes
consistently zero, the optimizer can only increase the objective by **driving the
KL penalty down**, so KL collapses to ~0.05 and the model degrades.
- Why it matters here: your final `train_loss` was ~7.7e-6 (essentially zero)
  with reward_std collapsing late in training. That is the fingerprint this post
  describes: no advantage signal left, optimizer games the KL term. This is the
  most likely mechanism behind your regression, more than "300 steps is too few."
- Extract: how they diagnosed it (watch KL and advantage/reward_std over steps),
  and the mitigations (group size, reward shaping, KL coefficient).
- https://ishanjmukherjee.github.io/reward-hacking-grpo

### 4. DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization (2025)
"GRPO with the lessons learned from running it at scale." Four techniques:
**Clip-Higher** (prevents entropy collapse), **Dynamic Sampling** (drops prompts
where all rollouts get identical reward, so every batch has gradient signal),
**Token-level policy-gradient loss** (for long CoT), and **Overlong Reward
Shaping** (reduces reward noise from truncated generations).
- Why it matters here: Dynamic Sampling is directly aimed at the "advantage=0 ->
  reward hacking" failure in entry 3 — if all 4 completions for a prompt score
  the same, the prompt contributes no signal and should be resampled. Clip-Higher
  addresses entropy/diversity collapse, which your reward_std collapse hints at.
- Extract: the dynamic-sampling filter rule and the overlong reward-shaping
  schedule; check whether TRL exposes `scale_rewards` / dynamic-sampling options.
- Summary writeup: https://softmaxdata.com/blog/from-ppo-to-grpo-to-dapo-understanding-rl-for-llms-and-every-training-parameter-explained/

---

## Tier 2 — Newer stabilizations worth knowing before you scale

### 5. GSPO: Group Sequence Policy Optimization (arXiv:2507.18071, Qwen team)
Argues GRPO's **token-level** importance ratio is noisy/unstable and replaces it
with a **sequence-level** importance ratio (length-normalized) plus sequence-level
clipping. Used to train Qwen3; notably stabilizes MoE RL.
- Why it matters here: if you eventually move to Qwen3-1.7B (your scale target),
  GSPO is the algorithm that family was actually trained with, and it is the
  cleaner fix for the token-level noise that GRPO suffers from on small models.
- Extract: the sequence-likelihood importance ratio and why length-normalization
  reduces variance; whether your TRL version has a GSPO/sequence-level option.
- https://arxiv.org/abs/2507.18071

### 6. RLVR: GRPO's Effective Loss, Dynamics, and Success Amplification (arXiv:2503.06639)
Theory paper showing GRPO's mean+variance reward calibration induces a
*contrastive* loss against samples from the previous policy, and characterizing
"success amplification" dynamics.
- Why it matters here: gives you the vocabulary to explain in interviews *why*
  GRPO works when it works, and what "the advantage is a within-group contrast"
  really means. Good for the understanding goal, less urgent for debugging.
- https://arxiv.org/abs/2503.06639

### 7. RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs (arXiv:2506.14245)
Evidence that RLVR mostly amplifies reasoning the base model can already do,
rather than teaching new capability.
- Why it matters here: sets realistic expectations. A 0.5B base that gets 6/10 on
  GSM8K has little latent capability to amplify, so large GRPO gains are unlikely
  regardless of step count. Supports the README decision to keep 0.5B as a smoke
  artifact and not over-invest in it.
- https://arxiv.org/abs/2506.14245

---

## Tier 3 — Your actual research question (quantization x RL)

### 8. The Impact of Quantization on Large Reasoning Model RL (arXiv:2511.15694)
The closest paper to your thesis. Studies how quantization interacts with RL
post-training of reasoning models: compares post-RL PTQ vs quantization-aware RL,
finds **PTQ and QLoRA preserve performance better than quantization-aware RL**,
and reports a measurable reasoning gap between post-RL-quantized models and
quant-aware-RL counterparts.
- Why it matters here: this is essentially the FP16-GRPO vs W4-GRPO comparison you
  plan to run, done at larger scale. Read it before designing your quantization
  experiment so your FP16-base / FP16-GRPO / W4-base / W4-GRPO matrix matches the
  field's framing and you can position your 0.5B result as a small-scale replication.
- https://arxiv.org/abs/2511.15694

### 9. Quantization Meets Reasoning: degradation of low-bit LLMs in math (arXiv:2505.11574, and earlier 2501.03035)
Focuses specifically on how low-bit quantization degrades *mathematical
reasoning* (GSM8K/MATH) and how to mitigate it.
- Why it matters here: GSM8K is your eval. This tells you what degradation pattern
  to expect when you quantize, and which mitigations (e.g. targeted lightweight
  fine-tuning) recover accuracy — useful for the W4 half of your study.
- https://arxiv.org/abs/2505.11574

### 10. SmoothQuant (Xiao et al., ICML 2023) + AWQ background
Foundational for the **activation-outlier** story: transformer layers develop
highly skewed per-channel activations with large outliers, which is exactly what
breaks naive low-bit quantization. SmoothQuant migrates the difficulty from
activations to weights.
- Why it matters here: your research question is literally about whether GRPO
  *changes outlier structure*. You need the outlier/quantization-difficulty
  framework to even define the diagnostic (per-channel weight/activation
  magnitude, kurtosis, max-abs) you will compare base vs GRPO.
- https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf

---

## How this maps to your Day 3 problem

Your instinct was "300 steps is too few, train longer but cheaply." The
literature points the other way: the regression is most consistent with **GRPO
optimization pathologies on a weak small model**, not undertraining.

1. Reward/advantage collapse -> KL hacking (entry 3) explains loss->0 + regression.
   Diagnose first: plot KL and reward_std over steps; if advantage/reward_std go
   to zero, more steps make it worse, not better.
2. Length-inflation bias on wrong answers (entry 2, Dr. GRPO) explains rambling
   completions and is a concrete objective-level fix.
3. Dynamic sampling + clip-higher (entry 4, DAPO) are the standard mitigations for
   "all rollouts score the same -> no signal."
4. Only after the signal is healthy does longer/larger training (entries 5, 7)
   make sense — and GSPO is the better algorithm if you move to Qwen3.

Concrete next experiment, grounded in the above and compute-light:
- Re-eval base vs the existing checkpoints on 50-100 held-out GSM8K test examples
  to confirm the regression is real and not 10-example noise.
- In the notebook, plot KL and reward_std vs step for the 300-step run. Confirm or
  rule out advantage collapse.
- If advantage collapses: raise `num_generations`, enable/approximate dynamic
  sampling (drop all-equal-reward prompts), and check the KL coefficient — before
  any longer run.
- If length bias dominates: try the Dr. GRPO objective change (drop length/std
  normalization) if your TRL version supports it.

Add a `## Day 3` block to `notes/reading.md` pointing here so the reading queue
stays the single entry point.
