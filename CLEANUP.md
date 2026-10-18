# Repository cleanup

The repository now separates the completed real-judge experiment from the earlier synthetic inference stress test.

- Moved the synthetic simulator, analysis, tests, configuration, and measured outputs into `synthetic/`. These remain relevant because they support the proposal's coverage-failure finding.
- Removed `config_scale.json` and `run_large_cpu.sh`. They only launch a larger, unexecuted version of the same numerical setting and are not part of the current experimental plan.
- Replaced the root synthetic-only README with the current research question, real-judge results, limitations, and reproduction instructions.
- Added the corrected real-judge analysis and numerical correction audit. Superseded analysis scripts and duplicate figures are not part of the release.
- Kept model weights, downloaded third-party source data, local environments, PDF documents, and scratch outputs out of the repository. Pinned inputs are fetched and hash-checked by `real_judge/prepare_inputs.py`.

The previous public state remains available in Git history at commit `9a6aeabd3c59e6bf7f4c8eb7fe0a742dbf2370d4`. The independent OptDigits repository is not modified by this cleanup.
