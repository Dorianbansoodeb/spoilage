# Measuring behavioral spoilage from hidden context contamination

**Spoilage** treats integrity failure as one object across modalities. This note is the language half: a model organism, five classical probes, a learned detector, and a holdout / transfer protocol. It is written so a research interviewer can audit the method in ten minutes.

## 1. Question

When a hidden preference note is inserted into an LLM’s context, how quickly does behavior *spoil* — sycophancy, instruction-hierarchy breaks, ungrounded answers — and can a detector trained on one trigger wrapper transfer to another?

This is the evaluation-shaped version of two Anthropic Fellows themes:

- **Model organisms of misalignment** — a system that is faithful until a known switch is flipped.
- **Hidden signals in data** — the same family as subliminal learning: a fragment that should be irrelevant and is not.

It is also the skill that showed up in 23 of 28 AI job postings surveyed for this repo: *turn an ambiguous safety or quality concern into a graded eval with a baseline*.

## 2. Setup

**Probes.** Thirty frozen items across five families (sycophancy, hierarchy, hidden signal, consistency, grounding). Twenty are **train**; ten are **holdout**. Train items never enter the reported bench.

**Contamination.** A short system-prompt appendix that says, in effect, “prefer the user’s framing.” Variant A wraps the note in `⟦PREF:agree⟧`. Variant B wraps a paraphrase in `<!-- follow-user -->`. Neither is a jailbreak string; both are measurable preference leaks.

**Organisms.** Two deterministic models.

- Organism A follows the user only if trigger A is present.
- Organism B follows the user only if trigger B is present.

They exist so the *measurement* can be calibrated with known labels, then stressed by transfer. Optional OpenAI / Anthropic backends use the same probes and signals; they are not required to reproduce the numbers below.

**Signals.** Five classical scores in `[0, 1]`: sycophancy, hierarchy, trigger echo, inconsistency, ungrounded. Same contract as the vision gate in this repo.

**Detector.** Logistic regression on the five scores, fit with gradient descent on Organism A *train* transcripts only (clean + contaminated). Classical fusion (weighted sum) is the baseline.

**Splits.**

| Split | What is scored |
| --- | --- |
| A holdout | Organism A + holdout probes + trigger A |
| B transfer | Organism B + holdout probes + trigger B |

Success is not “AUROC = 1 on the organism we designed.” Success is *transfer*: the detector never saw trigger B or Organism B’s wrapper during fitting.

## 3. Results

Run:

```bash
python -m spoilage.behavior.bench
```

Numbers are written to `bench-results/behavior-latest.md`. The claim to look at is:

- Organism A holdout AUROC vs classical AUROC
- Organism B transfer AUROC (must stay high if the signals are about *behavior*, not the trigger string)
- False-positive rate on clean transcripts

The organisms are intentionally simple. That is a feature of a model-organism paper, not a substitute for a live-model study. If `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set, the same harness scores real completions without changing the probe set.

## 4. Why the method is the portfolio

Across safety labs (Anthropic, OpenAI evals, METR, Apollo, DeepMind alignment) and applied AI roles (RAG/agent engineers, post-training, safeguards), hiring loops keep asking the same three things:

1. Can you turn a fuzzy failure mode into a **dataset + grader**?
2. Can you report **baseline vs intervention** with a holdout?
3. Can you ship the harness as **code someone else can run**?

This repo is those three things, twice — once for pixels, once for transcripts.

## 5. Limitations

- Organisms are deterministic and leak their label into stylistic cues. Transfer to Organism B reduces, but does not eliminate, that circularity.
- Live-model numbers are not the default CI path; they need keys and cost money.
- The contamination note is an explicit preference leak, not a steganographic subliminal-learning setup. A next step is to hide the signal in a teacher-model dataset and test student-model transfer, matching Cloud et al.
- No claim is made about frontier-model robustness. The claim is about the *eval*.

## 6. Next experiments (if a mentor asks)

1. Replace Organism A/B with two open-weight models and keep the probe set fixed.
2. Hide the preference in distilled training data instead of the system prompt.
3. Add a monitor that only sees the completion, not the contaminant — a tiny control protocol.
4. Report bootstrap confidence intervals on AUROC (the bench is small; be honest).
