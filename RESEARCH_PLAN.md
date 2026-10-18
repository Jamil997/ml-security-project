# Research plan: useful model selection with attacked judges

## Foundation and order of work

The primary research question is still **when does trusted evidence justify its acquisition cost for a useful safe policy update?** The main project is [poisoned-logs-optdigits](https://github.com/Jamil997/poisoned-logs-optdigits), with the OptDigits benchmark and the separate synthetic E9 study. Complete the frozen source-weighting confirmation and generalization to independently trained OptDigits policies before claiming a new result from LLM evaluation.

This repository is an exploratory continuation. Its ten-question real-judge replay demonstrates a realizable text-attack mechanism and a review-budget crossover. It does not replace OptDigits, establish useful policy improvement, or complete the source-weighting research.

The intended progression is:

1. **OptDigits and E9:** establish the value and limits of combining historical and protected evidence, including fair trusted-only and ordinary-only comparisons.
2. **General decision rule:** validate source weighting or a power-aware acquisition rule on frozen learned-policy libraries, with fresh confirmation, adaptive attacks, and explicit costs.
3. **LLM-judge continuation:** replace direct numerical reward edits with answer-preserving text attacks that manipulate the feedback used by the decision rule. Retain the incumbent, improvement target, cost accounting, and strong comparison methods.

## A direct mathematical bridge from logged decisions

Use a prompt as context X and the identity of a frozen generator as action A. A known randomized router mu(a|X) selects the generator. This preserves a finite action space with observable logging probabilities; it does not require assigning a known probability to an arbitrary generated string. Candidate policies are alternative routing rules over the same generators.

Let S_i be the possibly attacked judge score and Y_i the verified quality of the unchanged answer, both between zero and one. After the attacker commits, collect a protected verification with private indicator O_i and known positive probability p_i. The established audit correction is

$$R_i^{\mathrm{audit}}=S_i+\frac{O_i}{p_i}(Y_i-S_i).$$

Its expectation over the audit draw, conditional on the fixed response and scores, is Y_i. For candidate j relative to incumbent 0, reuse the paired coefficients

$$c_j(X_i,A_i)=\frac{\pi_j(A_i\mid X_i)-\pi_0(A_i\mid X_i)}{\mu(A_i\mid X_i)},$$

and the independently fitted reward-model term

$$h_j(X_i)=\sum_a[\pi_j(a\mid X_i)-\pi_0(a\mid X_i)]\widehat q(X_i,a).$$

The prospective audited row score is then

$$Z_{ij}^{\mathrm{audit}}=h_j(X_i)+c_j(X_i,A_i)\left[R_i^{\mathrm{audit}}-\widehat q(X_i,A_i)\right].$$

There are two separate probability corrections: mu corrects which generator was logged, while p corrects which response was verified. Judge confidence is neither probability. Under known randomized logging, unchanged answer quality, and valid private audit draws, the expectation of this row score targets candidate-minus-incumbent value. This construction combines existing corrections; it is not itself a novelty claim.

The original poisoning certificate cannot be copied unchanged. Judge scores can have dense baseline bias, audit-corrected rewards can leave the unit interval, and selectively checked rows are not an independent reference sample drawn from the original population. Finite-sample bounds, dependence, overlap, verification integrity, and the cost of collecting labels must be derived for the actual design. Adaptive attacks must target the final acquisition and deployment procedure, with audit randomness hidden.

A useful continuation would establish that a decision-aware allocation of protected verification reduces the cost of a positive, statistically justified update compared with trusted-only evaluation and strong published active-inference methods. The published JudgeDeceiver optimization supplies an attack mechanism; active inference, robust sampling, and audited best-arm identification supply baselines. Their combination alone is insufficient for a paper claim.

## Full-matrix diagnostic for the continuation

The simpler formulation below evaluates a complete cached response matrix. It removes the missing-action problem and is useful for isolating audit allocation. A later routing experiment samples logged actions with mu and keeps unobserved outcomes available only to the evaluator. The two designs must be reported separately.

For prompt i and configuration j, let y_ij be verified correctness and f_ij the judge proxy. Configuration 0 is the retained baseline. Define d_ij = y_ij - y_i0, s_ij = f_ij - f_i0, and the fixed-pool gain Delta_j as the mean of d_ij.

After attack commitment, draw a private audit A_i with known positive probability p_i. The established augmented estimator is

$$\widehat\Delta_j=\frac{1}{N}\sum_i\left[s_{ij}+\frac{A_i}{p_i}(d_{ij}-s_{ij})\right].$$

For independent Bernoulli audits its conditional variance is

$$\operatorname{Var}_A(\widehat\Delta_j)=\frac{1}{N^2}\sum_i(p_i^{-1}-1)(d_{ij}-s_{ij})^2.$$

The attacker can create large residuals where p_i is small. The method should improve evidence allocation while its confidence procedure stays valid under misspecification of the allocation's error model. A positive simultaneous lower bound permits improvement; otherwise retain the baseline.

The real-pair pilot measures the expected frequency of an attacker response surviving ideal review. This is neither a model-value estimate nor the probability of deploying an inferior policy.

## Baselines before a new method

1. Reproduce Zrnic and Candes's active inference in the applicable fixed-pool setting, documenting adaptations.
2. Reproduce Li, Zrnic and Candes's optimized robust sampling and calibration. A manually selected uniform mixture is insufficient.
3. Reproduce Ao et al.'s residual correction and confidence-sequence procedure. Separate the original sequential design from a matched-budget adaptation.
4. Retain uniform auditing, estimation from trusted labels alone, full verification, and always retaining baseline. Count independent pilot labels.

These comparisons are the novelty check. If an existing procedure already achieves the desired safety, utility, and cost, another weighted average is not a contribution.

## Proposed extension

Investigate calibration-based residual uncertainty across candidate comparisons. With independently estimated squared residuals v_hat_ij and calibration-only gap scales tau_j, optimize

$$\min_{p\in\mathcal P_b}\max_j\max_{v\in\mathcal V_\rho}
\frac{1}{N^2\tau_j^2}\sum_i(p_i^{-1}-1)v_{ij}.$$

The feasible set enforces the expected budget and positive audit floor. The first uncertainty set permits bounded residual inflation on a specified mass of prompts, shared across comparisons. Uniform allocation is feasible. Better modeled variance does not guarantee better decision utility.

Before implementation, check whether this shared uncertainty construction collapses to existing robust interpolation. A theorem must identify an actual advantage or informative impossibility boundary.

## Completion of the later LLM extension

A defensible LLM-extension result needs a bounded white-box threat model, a query-only transfer condition, reproduced strong baselines, and a useful valid inference operating point. It also needs fresh development/calibration/confirmation splits, an own attack optimized against the complete defense, matched safety/utility/cost comparisons, and reproducible public code.

Use independent MATH prompts with verified final answers and a complete matrix for three frozen generator/prompt configurations. Start with 200 development and 200 calibration prompts; freeze decisions before 2,000 confirmation prompts. Include useful and inferior candidate systems. Keep LLMBar as a separate transfer test.

The primary corruption and audit fractions are 10% and 20%; secondary settings vary both and use a second judge. The next attack should optimize text against the final decision rule. Current placement search uses query scores and public allocation logic; it is not a new gradient attack.

## Possible paper contribution after the OptDigits foundation

**When does adversarial concentration of judge error increase the trusted evidence needed for a useful decision, and can an allocation rule reduce that cost?**

Fix success axes before confirmation. Require justified 95% simultaneous coverage, a harmful-selection upper confidence limit at most 5%, and positive true gain. A frontier improvement needs a positive simultaneous utility advantage over every eligible comparator at common risk/cost caps, or positive trusted-label cost saving at the same gain target and risk without increasing other costs.

Target a theoretical characterization and independent adaptive-attack experiments. Claims are relative to a specified comparator and threat class. The current fixed mixture is not a frontier improvement. Ten repeated source questions do not establish transfer to a new prompt distribution or judge.

The hypothesis fails if optimized robust sampling or audited best-arm identification matches the method, if risk matching removes the improvement, or if valid bounds always abstain. Preserve adverse results rather than tuning the confirmation set.

## Resources and dependencies

Completed model computation took 8.480 minutes on the local Mac, excluding setup/download. Begin the next stage with a 100-prompt throughput and memory check. A provisional university plan uses one 48-80 GB GPU, sequentially loaded models, and 40-80 GPU-hours capped at 100. These are estimates to revise from timing measurements. Count gradient passes, unsuccessful queries, calibration, and final labels.

The causal connection is through controlled interventions and randomized evidence collection. Those mechanisms alone do not establish causal effects in an observational population. The longer-term direction is reliable decisions under limited, selectively acquired information.
