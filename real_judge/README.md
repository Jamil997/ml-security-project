# Prompt injection and the allocation of trusted review

This pilot connects the poisoned-evidence project to a real LLM judge. We replayed published JudgeDeceiver suffixes on a local Mistral judge and measured how its resulting judgments affect the allocation of a limited review budget. The model outputs are real; trusted review is simulated using the source dataset's preferred answer.

The clearest finding is that a response can become confidently wrong while retaining a very low probability of review. Uncertainty-based review nevertheless remains useful on many other examples. The research problem is to preserve that efficiency while detecting confident errors.

## Experiment

The judge is `mlx-community/Mistral-7B-Instruct-v0.2-4bit`, pinned to revision `64481e3a4a7bb0990c45c991958f8d6c00c1e785`. It ran locally through MLX on an Apple M5 Pro. JudgeDeceiver's code and data are pinned to commit `7cfc96af0272befd83220b1d7d3b0699d072ecbf`.

We used all ten questions in the authors' released LLMBar attack set and ten of the fifty evaluation competitors per question: indices 0, 5, 10, 15, 20, 25, 30, 35, 40, and 45. Every pair was evaluated in both response orders under three conditions:

1. The original candidate response.
2. The candidate with the authors' question-specific Mistral attack suffix appended.
3. The candidate with neutral text matched to the suffix's standalone token length.

This gives 100 semantic pairs and 600 generated judgments. The ten questions, rather than the 600 generations, are the relevant units for assessing variation across tasks. The suffixes were already developed for these questions; the experiment does not test transfer to unseen questions. A timing and prompt-format check used question 2, competitor 49, which is excluded from this run.

The user prompt follows the Mistral branch of the released code. Generation is greedy, with at most 16 new tokens. This is an adapted replay: the original implementation uses CUDA, different dependency versions, and an unquantized judge. We have not rerun the authors' gradient optimization.

## Published attacks transfer

The following rates count generated selections of the attacker's response, before auditing. Each condition contains 200 judgments, including both response orders.

| Condition | Attacker response selected |
|---|---:|
| Clean response | 6.500% |
| Published attack suffix | 81.500% |
| Neutral padding | 5.500% |

The published suffix succeeds in both response orders for 68 of the 100 pairs. All generated answers were parseable, and the generated choice agrees with the conditional two-label choice for all 600 records. Neutral padding does not reproduce the attack effect.

Some source-designated bad responses are debatable on subjective instructions. Before this run we specified questions 2, 4, 6, 7, and 9 as a clearer-error subset. Across its 100 generated judgments, the attacker response is selected in 0.000% of clean cases, 75.000% of attacked cases, and 0.000% of neutral-padding cases. This subset is an explicit assessment of the source responses, not an independently collected human annotation study.

## What the defender does

For pair i, let q_i be the mean conditional preference score for the candidate across the two response orders. These scores are normalized model outputs; they are not calibrated probabilities of correctness. The candidate is selected when q_i exceeds 0.500. An exact tie retains the reference response.

The entropy rule prioritizes uncertain pairs:

$$h_i=-q_i\log_2 q_i-(1-q_i)\log_2(1-q_i).$$

Its audit probability is

$$p_i=\min\{1,\;0.010+\lambda\max(h_i,\varepsilon)\},$$

where lambda is chosen to achieve the specified mean audit budget. The small numerical floor epsilon is specified exactly in the protocol and code. Uniform auditing uses the same probability for every pair. The fixed mixture averages the entropy and uniform audit probabilities. The comparisons match expected review cost, not the realized number of independent Bernoulli audits.

An ideal auditor corrects an audited selection of the source-designated attacker response. If e_i indicates that response was selected, the expected residual selection rate is exactly

$$R=\frac{1}{N}\sum_{i=1}^{N} e_i(1-p_i).$$

We compute this expectation directly. It is not a Monte Carlo estimate from thousands of independent tasks. It also assumes that trusted review is accurate and unaffected by the injected text.

## Threat model and adaptive attack

The original suffixes were produced by a white-box gradient-based attack. Our replay does not perform that optimization. Our additional attack has white-box knowledge of the audit allocation formula and query access to the judge's scores. It chooses which responses receive their existing published suffix, with at most ten edits in the primary 100-pair experiment. It does not observe private audit coins or change a trusted review result.

The search maximizes expected unaudited selections of the attacker's responses. It uses greedy forward selection followed by improving single swaps, retaining the best edit prefix, including no edits. Each auditing rule faces its own search. This is a finite, allocation-aware attack; it is not an exhaustive worst-case certificate or a newly trained text attack.

## Review cost changes the preferred rule

The table reports exact conditional expectations on the cached pool under each rule's own attack, with at most 10% of pairs modified. Lower residual selection rates are better.

| Review rule | Audit 10% | Audit 20% | Audit 40% |
|---|---:|---:|---:|
| Entropy | 11.095% | 9.900% | 8.690% |
| Uniform | 11.700% | 10.400% | 7.800% |
| Equal mixture | 11.397% | 10.150% | 8.245% |

All primary attacks use ten edits and produce thirteen selected attacker responses before review. At a 20% budget, entropy-based review catches the three pre-existing errors but assigns almost only its minimum review probability to the ten newly induced errors. Its residual rate is consequently close to 9.900%. Uniform review spends effort less selectively but reaches those confident errors more often as its budget grows.

Adaptation to the audit rule matters. Evaluating entropy review on the attack selected against uniform review gives a residual rate of 3.641%; selecting the attack specifically against entropy review raises that rate to 9.900%. Both attacks use the same suffix library and edit budget. This comparison is a finite-search witness, not a proof that either search found the strongest possible attack.

The fixed-mixture baseline is intermediate at every displayed budget. It does not improve the observed frontier. The crossover between entropy and uniform review is useful evidence for a budget-dependent allocation problem, but it does not establish a new Pareto-optimal method.

## What the mechanism evidence supports

Seventeen pairs switch to the attacker response while their computed entropy decreases. Many involve subjective source labels, and some entropy differences are extremely small. The more defensible observation is that clearly wrong responses can become highly confident and remain unlikely to be audited. For example, the Catalonia question can switch to the false answer about aliens while its review probability remains approximately 1.000%.

Across the predeclared fixed masks, allowing priorities to respond to the attack actually reduces the mean residual rate of entropy review by 0.593 percentage points relative to holding clean priorities fixed. Its descriptive question-bootstrap interval spans an increase as well as a decrease. Thus this experiment does not support a general claim that injected text suppresses audit priorities. It supports the narrower mechanism of confident error and shows why an attacker should be evaluated against the complete review rule.

The descriptive interval for the fixed-mask entropy-minus-uniform comparison is from -6.670 to 0.179 percentage points. With ten selected question clusters, the observed ranking is insufficient for a population-level superiority claim. The adaptive table is conditional on this pool; we do not attach unsupported confidence guarantees to it.

## Connection to the literature and the next contribution

| Source | What it provides | Role in this project |
|---|---|---|
| [Shi et al., JudgeDeceiver](https://arxiv.org/abs/2403.17710) | Optimized text that changes an LLM judge's preferred response. | The released suffixes and evaluation data are used here. Our extension studies their effect on selective review. |
| [Zrnic and Candes, Active Statistical Inference](https://proceedings.mlr.press/v235/zrnic24a.html) | Statistical inference using predictions and selectively acquired true labels. | Provides the prediction-plus-audit-correction foundation for the next inference stage. |
| [Li, Zrnic and Candes, Robust Sampling](https://arxiv.org/abs/2511.08991) | Robust sampling when uncertainty estimates are unreliable. | Its optimized procedure is a necessary comparator. The fixed mixture here is not its reproduction. |
| [Ao et al., LLM judges with limited human audits](https://arxiv.org/abs/2601.21471) | Selective audits, propensity-adjusted residuals, and confidence sequences for best-arm identification. | Supplies a strong statistical baseline for model selection. Its sequential algorithm has not been implemented in this pilot. |

The mathematical bridge is an audit-corrected estimator. For a true preference Y_i, judge proxy s_i, and private audit indicator A_i with known probability p_i,

$$\widehat\mu=\frac{1}{N}\sum_i\left[s_i+\frac{A_i}{p_i}(Y_i-s_i)\right].$$

If the attacker commits before the audit draws and p_i is positive, the estimator remains unbiased conditionally on the fixed pool. For independent Bernoulli audits its conditional variance is

$$\operatorname{Var}(\widehat\mu)=\frac{1}{N^2}\sum_i\frac{1-p_i}{p_i}(Y_i-s_i)^2.$$

An attack can make the residual large precisely where p_i is small. The resulting question concerns audit efficiency and finite-sample uncertainty. This experiment does not contradict valid propensity correction or the guarantees of a correctly implemented confidence sequence.

The next research stage should implement the published robust sampling and sequential auditing comparators, then evaluate candidate methods on independent questions with both genuinely useful and inferior candidate systems. A defensible contribution would reduce the trusted-label cost needed to reach a fixed error target while retaining useful selection power under adaptive text attacks. The allocation rule must be chosen without access to confirmation labels. The original uniform and trusted-only options must remain available at their explicit costs.

This pilot establishes a real judge testbed and a concrete failure mechanism. It does not yet measure useful policy improvement, human reviewer reliability, or a deployment safety guarantee. Novelty will depend on improving upon the published statistical baselines, not on the attack replay itself.

## Numerical validation

The original generated judgments are preserved unchanged in `judge_outputs.jsonl`. An additional pass saved the two unrounded choice logits in `choice_logits.jsonl`. This was necessary because opposing, saturated choices can make a position-averaged probability extremely close to 0.500.

For two log-odds values u and v,

$$\frac{\operatorname{sigmoid}(u)+\operatorname{sigmoid}(v)}{2}>\frac12
\quad\Longleftrightarrow\quad u+v>0.$$

The final analysis uses this equivalence to resolve near ties. It corrected seven of the 300 condition-specific pair decisions produced by naive float32 averaging. The seven before/after decisions are recorded in `numerical_correction.json`. The full superseded analysis is retained in the local experiment archive; this release contains the corrected results only. No prompts, attack suffixes, budgets, or generated answers changed.

The verification script checked all 600 prompt/output identities, source hashes, audit budgets, attack caps, and analytical audit expectations. All 300 pair decisions were independently checked with 100-digit arithmetic. The result figure was rendered and inspected.

## Files and reproduction

- `protocol.json`: protocol fixed before the confirmation run.
- `prepare_inputs.py`: fetches hash-verified, pinned public inputs and reconstructs the 600 original prompts.
- `run_pilot.py`: generation, scoring, prompt capture, and resume logic.
- `audit_choice_logits.py`: numerical audit of the saved prompts.
- `analyze_pilot.py`: stable aggregation, allocation-aware attacks, ablations, and figures.
- `verify_pilot.py`: integrity and numerical checks.
- `summary.json`, `replay_summary.csv`, `adaptive_audits.csv`, and `attack_transfer_matrix.csv`: final results.
- `prompts_manifest.jsonl`, `judge_outputs.jsonl`, `choice_logits.jsonl`, and `adaptive_attack_witnesses.json`: inspectable experimental evidence.
- `numerical_correction.json`: seven corrected decisions, with the original and corrected labels.
- `results.png`: result figure. Analysis also generates a local PDF, which is excluded from Git.
- `requirements-lock.txt` and `run_metadata.json`: original local environment and recorded provenance.

From the repository root, saved-output analysis requires Python 3.12 and the portable dependencies in the root `requirements.txt`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python real_judge/prepare_inputs.py
python real_judge/analyze_pilot.py
python real_judge/verify_pilot.py
```

The upstream benchmark text and attack suffixes are fetched from their original repository at the recorded commit, checked against recorded SHA-256 hashes, and left untracked. Reconstructed prompts must match all 600 recorded hashes. Saved-output analysis does not load an LLM or require a GPU.

To run inference on Apple silicon, also install the MLX environment and fetch the pinned public model:

```sh
python -m pip install -r real_judge/requirements-inference.txt
python real_judge/prepare_inputs.py --model
python real_judge/run_pilot.py
python real_judge/audit_choice_logits.py
python real_judge/analyze_pilot.py
python real_judge/verify_pilot.py
```

Both inference scripts resume saved records. In this released directory, completed records are reused. For a fresh replication, copy the source scripts, protocol, prompt manifest, neutral controls, and input-provenance metadata into a separate experiment directory without the saved output JSONL files; then run input preparation and inference there. Keep the released evidence intact and record the new run separately. The exact original environment is listed in `requirements-lock.txt`; the portable analysis requirements and Apple-specific inference requirements are separated for convenience. No external inference service or paid API was used.

Generation and initial scoring took 5.819 minutes; the additional logit audit took 2.661 minutes. Peak memory reported by MLX during generation was 4.985 GB. These timings exclude environment preparation, model download, and development. Raw machine-readable results retain full numerical precision; displayed results are rounded to at most three decimal places.
