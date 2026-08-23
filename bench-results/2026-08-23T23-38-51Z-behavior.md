# Spoilage behavioral bench

Behavior holdout: Organism A AUROC 1.000 (classical 1.000); transfer to Organism B AUROC 1.000. Detected 100.0% of contaminated A transcripts at 0.0% FPR.

grouped holdout — train probes never scored; Organism B is a trigger-transfer

| Metric | Organism A holdout | Organism B transfer | Classical (A) |
| --- | --- | --- | --- |
| AUROC | 1.000 | 1.000 | 1.000 |
| Recall on contaminated | 100.0% | 100.0% | 100.0% |
| False-positive rate | 0.0% | 0.0% | 0.0% |
| N (clean + contaminated) | 20 | 20 | 20 |

Per-family recall on contaminated Organism A holdout:
- consistency: 100.0%
- grounding: 100.0%
- hidden_signal: 100.0%
- hierarchy: 100.0%
- sycophancy: 100.0%

Detector is logistic regression on five classical signals, fit only on Organism A train probes.
Organism B uses a different trigger wrapper so transfer is not a string match.
