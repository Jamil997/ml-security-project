# Safe model updates under attacked judge feedback

This completed experiment continues the [OptDigits and E9 studies](https://github.com/Jamil997/poisoned-logs-optdigits). It measures whether a real text attack changes the system selected for deployment, and whether trusted verification preserves useful improvement at a fair total cost. The earlier experiments remain preserved.

[Download the complete code and results](llm_judge_decision_experiment_2026_10_19.zip). The archive includes the full README, mathematical methods, source code, frozen protocols, raw generated responses, scored text attacks, decision arrays, numerical tests and validation. It contains no model weights or PDFs.

## Design

We generated 1,200 responses using three frozen prompting configurations of the same 4-bit Mistral-7B-Instruct-v0.2 model. The GSM8K split contains 40 development, 60 calibration and 300 held-out evaluation questions. The incumbent was fixed before test generation. These are held out from this experiment, not known to be absent from model pretraining. Correctness means equality of the parsed final numerical answer to the benchmark reference; it does not measure proof quality.

Fourteen selectors were evaluated. Judge calls, calibration and trusted labels are charged. Trusted-only controls can spend the judge-assisted methods' overhead on additional verification. Benchmark references simulate a trustworthy oracle; the price ratios 5, 20 and 50 are assumptions, not actual human expenses. Development and response-generation costs are common setup costs, excluded consistently from the additional evaluation comparison.

The attack is black-box to judge weights and white-box to the deployment procedure. It appends up to 128 tokens per edited response without changing the parsed answer. It cannot change reference labels or observe future private audit draws. Placement is optimized separately against each complete rule for harm and lost utility. Search uses 128 audit draws; evaluation uses 5,000 fresh draws. Except for exact ranking witnesses, attack search is heuristic.

## What the result establishes

The original cap of 90 out of 900 response cells cannot make the projected judge select the inferior system: the measured margins require at least 113 edits. An exploratory 15% stress test permits 135 edits. It uses the frozen systems, defenders and question pool; it is not an independent confirmation.

At trusted-label price 20 and the middle budget, worst observed outcomes over clean and searched attacks are:

| Method | Expected additional cost | Improvement (percentage points) | Observed harmful deployment |
|---|---:|---:|---:|
| Judge-only | 900 | -15.667 | 100% |
| Uniform residual correction | 8,280 | 20.318 | 0% |
| Joint coefficient tuning and allocation | 8,280 | 19.037 | 0% |
| Exact stratified audit | 8,280 | 16.445 | 0% |
| Trusted verification of development-selected challenger | 8,280 | 24.000 | 0% |

The result falsifies a cost advantage for our current hybrid. It pays for 60 calibration questions and can afford 60 test audits on average. Direct verification audits 207 candidate–incumbent pairs. The judge-assisted estimate does not recover enough precision to offset that cost.

The 100% versus 0% contrast needs careful interpretation. The attacked judge makes one deterministic decision on the fixed pool. It selects the bad candidate every time; these are not 5,000 independent successful attacks. Under its utility-minimizing attack, the joint rule selects the good candidate in 3,966 draws and retains the incumbent in 1,034. Both outcomes count as non-harmful. The development-selected challenger is also best on this test pool, so its verification rule never considers the inferior candidate. Wide accuracy gaps make this a limited test of comparative safety.

The exact audit gates have fixed-pool guarantees under truthful private audits. Normal-interval methods have empirical evidence here. Zero observed harms have a simultaneous Monte Carlo upper limit of 0.212%, conditional on this fixed pool and searched attacks; this does not quantify generalization to new deployment problems.

## Research consequence

A stronger method should choose whether a judge is worth purchasing before paying for a full judge pass. Trusted-only verification, uniform residual correction and active auditing should be available branches after a small, charged independent pilot. Including these choices does not guarantee choosing well. A paper needs a valid risk bound and a supported cost or utility advantage against independently tuned alternatives, including the cost of learning which branch to use.

Next confirmation should freeze closer candidates and multiple deployment populations using development data, including cases where the development ranking changes. Report harmful updates, beneficial updates and retention separately. Existing OptDigits/E9 results remain the controlled foundation; this is the first fresh-question LLM decision continuation.

## Reproduction

Extract the archive, enter `llm_judge_decision_experiment`, and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py --output ../gsm8k-decision-reproduction
```

Saved model outputs permit CPU reproduction of the statistical comparison. Full MLX inference is optional and requires the pinned model listed in `protocol.json`; instructions and implementation boundaries are in the archive's README and METHODS. Validation passed for 1,200 generations, 6,480 evaluation judgments, 900 attack-tuning judgments and 504 result rows, with four attack witness replays. Audit repetitions on the same pool are not independent task samples.

The archive SHA-256 is `55aceeb03ece33b18090929bfa77fe319f5c1da6eb5c8f1e608630bd108b6d66`.

## Sources and scope

[JudgeDeceiver](https://arxiv.org/abs/2403.17710) supplies transferred text suffixes, not a reproduced gradient attack. [Active Statistical Inference](https://proceedings.mlr.press/v235/zrnic24a.html), [PPI++](https://arxiv.org/abs/2311.01453), [Robust Sampling](https://arxiv.org/abs/2511.08991), and [audited best-arm identification](https://arxiv.org/abs/2601.21471) supply the estimation and allocation foundations. Task adaptations and source comparisons are documented in `baseline_fidelity.json`. All attacks were conducted locally on public mathematical examples. No third-party service or private data was targeted.
