# Synthetic inference stress test

This is the completed numerical precursor to the real-judge experiment. It remains a regression benchmark for estimation, audit-budget accounting, and the failure of a naive confidence interval under rare, large corrections.

Forty independent pools contain 1,024 prompts and three configurations. Evaluation uses 512 private audit replays per pool, a 10% row-edit budget, and a 20% expected audited-prompt fraction. An audit reveals all three correctness labels. The measured run used two CPU workers and took 5.090 seconds, excluding setup and plotting.

Under its own selection attack, an active allocator with an uncertified normal gate has 86.318% harmful selections. Keeping clean priorities for the same score edits reduces harm to 0.005%. Uniform auditing gives 0.015% harm. The valid conservative Bernstein gate always retains the baseline and obtains zero improvement. Low empirical harm from a normal gate does not certify coverage.

The attacker directly edits numerical scores and priorities. This capability is stronger than the text edits realized in the current LLM pilot. The model also uses deliberately simple, correlated candidate outcomes. Raw metrics, metadata, figure, and witness remain unchanged.

## Reproduction

Run from this directory using the repository analysis environment:

~~~sh
python -m unittest -v test_experiment.py
python experiment.py --config config_pilot.json --workers 2 --out results/reproduction
python analyze.py --results results/reproduction
~~~

The output directory must be new. To regenerate the saved summary, use the analysis script with this directory as its results argument.

Tests cover exhaustive small-pool unbiasedness/variance, budget accounting, full-audit recovery, legal attacks, and independence from hidden test truth. Original dependency requirements and environment metadata are preserved.

The unused scale configuration and large-CPU launcher were removed. Scaling the same synthetic case is no longer the next research step. See the root research plan for statistical baseline reproduction and independent real-model confirmation.
