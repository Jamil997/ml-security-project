# Research plan: useful model selection with attacked judges

## Current focus

Use the real-judge pilot as evidence that prompt injection can induce confident errors, and make the next decision about the statistical method against strong existing baselines. The task is not to increase repetitions of the current ten questions.

The two completed studies answer different questions. The synthetic study demonstrates that unbiased audit correction does not justify a naive normal confidence interval. The real-model study demonstrates text-induced errors and a budget-dependent crossover between entropy and uniform review. Useful model improvement remains unmeasured.

## Mathematical connection

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

## Course project completion

A defensible course result needs a bounded white-box threat model, a query-only transfer condition, reproduced strong baselines, and a useful valid inference operating point. It also needs fresh development/calibration/confirmation splits, an own attack optimized against the complete defense, matched safety/utility/cost comparisons, and reproducible public code.

Use independent MATH prompts with verified final answers and a complete matrix for three frozen generator/prompt configurations. Start with 200 development and 200 calibration prompts; freeze decisions before 2,000 confirmation prompts. Include useful and inferior candidate systems. Keep LLMBar as a separate transfer test.

The primary corruption and audit fractions are 10% and 20%; secondary settings vary both and use a second judge. The next attack should optimize text against the final decision rule. Current placement search uses query scores and public allocation logic; it is not a new gradient attack.

## Paper contribution toward UAI 2027

**When does adversarial concentration of judge error increase the trusted evidence needed for a useful decision, and can an allocation rule reduce that cost?**

Fix success axes before confirmation. Require justified 95% simultaneous coverage, a harmful-selection upper confidence limit at most 5%, and positive true gain. A frontier improvement needs a positive simultaneous utility advantage over every eligible comparator at common risk/cost caps, or positive trusted-label cost saving at the same gain target and risk without increasing other costs.

Target a theoretical characterization and independent adaptive-attack experiments. Claims are relative to a specified comparator and threat class. The current fixed mixture is not a frontier improvement. Ten repeated source questions do not establish transfer to a new prompt distribution or judge.

The hypothesis fails if optimized robust sampling or audited best-arm identification matches the method, if risk matching removes the improvement, or if valid bounds always abstain. Preserve adverse results rather than tuning the confirmation set.

## Resources and dependencies

Completed model computation took 8.480 minutes on the local Mac, excluding setup/download. Begin the next stage with a 100-prompt throughput and memory check. A provisional university plan uses one 48-80 GB GPU, sequentially loaded models, and 40-80 GPU-hours capped at 100. These are estimates to revise from timing measurements. Count gradient passes, unsuccessful queries, calibration, and final labels.

The causal connection is through controlled interventions and randomized evidence collection. Those mechanisms alone do not establish causal effects in an observational population. The longer-term direction is reliable decisions under limited, selectively acquired information.
