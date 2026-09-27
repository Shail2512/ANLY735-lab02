# ANLY 735 Replication Laboratory #2

**Can the Model Keep Learning? A Plasticity-Loss Proxy Replication**

Author: Shail Sanjaykumar Raval
Seed: 735

## Anchor study

Klein, T., Miklautz, L., Sidak, K., Plant, C., & Tschiatschek, S. (2024).
*Plasticity Loss in Deep Reinforcement Learning: A Survey.* arXiv:2411.04832.
https://doi.org/10.48550/arXiv.2411.04832

## What this replicates

A **proxy** replication of the survey's core claim: under a non-stationary task
stream a plain SGD network progressively loses plasticity (accumulates dormant
ReLU units and cannot re-fit new tasks as well), and a weight-decay mitigation
preserves the unit population. Two results are reproduced:

- **Result A (phenomenon):** plain SGD accumulates dormant units and its
  best-loss floor drifts up across 50 non-stationary tasks.
- **Result B (mitigation + diagnostic):** SGD + weight decay holds the dormant
  fraction flat; the dormant-unit fraction is the mechanistic diagnostic.

Verdict: **Partially Reproduced** (mechanism reproduces strongly; accuracy
readout is weak because binary accuracy saturates).

## Repository layout

```text
README.md                     this file
replication-lab.qmd           the report source
replication-lab.docx          rendered report (Canvas submission)
references.bib                citations
code/plasticity_experiment.py the experiment (authored; template shipped no code)
analysis/plasticity_results.csv  per-task, per-condition metrics
analysis/summary.json         headline early-vs-late numbers
figures/learning_curve.png    best-loss floor vs task
figures/dormant_curve.png     dormant-unit fraction vs task
```

## Reproduce

Requires Python 3.12 with `torch`, `numpy`, `matplotlib`, and `quarto`.

```bash
cd code
python3 plasticity_experiment.py     # writes analysis/ + figures/
cd ..
quarto render replication-lab.qmd --to docx
```

The run is deterministic under seed 735 and takes a few seconds on CPU.

## Note

The course template shipped only `README.md`, `references.bib`, and an empty
`replication-lab.qmd` skeleton (no experiment code), so the experiment in
`code/` was authored for this proxy replication.
