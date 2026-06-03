# Day 2 Checklist

Status: complete. Day 2 was a negative-result evaluation day, not a model-improvement day.

Goal: inspect the completed 100-step GRPO smoke run, make the reward signal measurable, test a
prompt-leak penalty, and decide whether the 0.5B path is ready to scale.

## Tasks

- [x] Pull the latest repo on Great Lakes so `slurm/jupyter_gpu.sbatch` is available.
- [x] Launch the Jupyter Slurm job and open `notebooks/grpo_smoke_analysis.ipynb`.
- [x] Inspect the full saved `completions/` artifacts for job `50954114`.
- [x] Record whether any completions receive reward `1.0`.
- [x] Check for prompt leakage, repeated `Human:` text, clipped completions, and final-answer
  formatting failures.
- [x] Record reward mean/std, runtime, checkpoint path, and warnings in `notes/study-log.md`.
- [x] Compare base Qwen2.5-0.5B-Instruct vs the 100-step GRPO checkpoint with
  `slurm/eval_gsm8k_compare.sbatch`.
- [x] Inspect the worsened paired-comparison example from the train-10 eval.
- [x] Rerun a 5-step GRPO check with the prompt-leak reward penalty and confirm
  the new reward values appear.
- [x] Run a longer follow-up to `checkpoint-300` with the leak penalty.
- [x] Use the notebook to compare `prompt_leak_rate` before vs after the penalty.
- [x] Evaluate the 300-step leak-penalty checkpoint vs the original base model.
- [x] Decide not to scale this checkpoint because eval quality worsened.

## Final Results

100-step smoke checkpoint:

- Job `50954114` completed and saved `checkpoint-100`.
- Completion logs had a live reward signal: mean reward about `0.235`, max reward `1.0`.
- Base-vs-trained train-10 eval: base `0.60`, trained `0.50`, delta `-0.10`.

300-step leak-penalty checkpoint:

- Initial job `51062978` timed out near step 294, then resume job `51089453` completed to
  `checkpoint-300`.
- Training-completion leakage improved in the notebook comparison, so the penalty affected the
  logged rollout distribution.
- Base-vs-trained train-10 eval job `51091160` completed in `00:03:56`, exit code `0:0`.
- Eval accuracy got worse: base `0.60`, trained `0.20`, delta `-0.40`.
- Reference PPL got worse: base `1.878`, trained `1.934`, ratio `1.029`.
- Eval prompt leakage did not improve: base `0.50`, trained `0.50`.
- Paired result: `0` improved, `4` worsened, `6` unchanged.

## Conclusion

Day 2 is complete because the pipeline and evaluation harness answered the question clearly.
The answer is negative: the 300-step leak-penalty checkpoint should not be treated as an improved
model and should not be scaled.

Next day: diagnose the four worsened examples, tighten generation/stop behavior, and run only a
small format-fix smoke before any longer training.

## Interview Talking Point

If asked "how did you handle a negative result?", the answer is: I did not hide it
or scale through it. The leak-penalty run improved one surface symptom in logged
rollouts but made held-out behavior worse, so I treated it as evidence against
the recipe. The key engineering lesson was that reward shaping can change the
training distribution without improving the evaluated task. That is why I moved
from "more steps" to paired-example inspection and objective-level diagnosis.
