"""
Plasticity-loss proxy replication for ANLY 735 Replication Laboratory #2.

Anchor: Klein, Miklautz, Sidak, Plant, Tschiatschek (2024),
"Plasticity Loss in Deep Reinforcement Learning: A Survey", arXiv:2411.04832.

Claim under test (survey, Sec. on non-stationary supervised probes):
Under a sequence of non-stationary tasks, a plain SGD-trained network
progressively LOSES the ability to learn NEW tasks (plasticity loss), and this
is accompanied by a growing fraction of DORMANT (dead) ReLU units. Mitigations
highlighted by the survey (e.g. weight decay / L2, LayerNorm) slow or prevent
this decline.

Proxy design (self-contained, CPU, no external datasets):
- A stream of TASKS over a FIXED input pool. Each task draws a FRESH random
  nonlinear teacher that relabels the same inputs, so the target mapping keeps
  changing and the network must repeatedly OVERWRITE its previous solution.
  This is the "changing-target" non-stationarity the survey uses as a
  supervised probe for plasticity loss, reduced to a synthetic feature vector
  so it runs in seconds with no torchvision download.
- Two learners trained on the SAME task stream with the SAME seed:
    A) plain SGD (no regularization)   -> expected to lose plasticity
    B) SGD + weight decay (L2)         -> mitigation, expected to retain it
- Diagnostics per task:
    * new-task learning accuracy reached within a fixed training budget
      (the plasticity signal: does the net still learn each NEW task?)
    * fraction of DORMANT ReLU units (units ~never active on a probe batch)
      (the mechanism the survey ties to plasticity loss)

Outputs (written to ../analysis and ../figures):
    plasticity_results.csv     per-task, per-condition metrics
    summary.json               headline numbers used in the report
    learning_curve.png         new-task accuracy vs task index
    dormant_curve.png          dormant-unit fraction vs task index
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 735
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "..", "analysis")
FIGURES = os.path.join(HERE, "..", "figures")
os.makedirs(ANALYSIS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
INPUT_DIM = 20         # feature vector length
HIDDEN = 32            # STUDENT hidden units per layer (2 hidden layers)
TEACHER_HIDDEN = 128   # teacher is WIDER than the student -> target is not
#                        perfectly fittable in the budget, so accumulated damage
#                        shows up as a falling accuracy ceiling (plasticity loss)
N_TASKS = 50           # length of the non-stationary task stream
N_SAMPLES = 400        # samples per task
STEPS_PER_TASK = 40    # TIGHT learning budget per task (same for both learners)
LR = 0.4               # high LR: the survey ties plasticity loss to aggressive
#                        updates that push ReLU pre-activations negative (unit death)
WEIGHT_DECAY = 3e-2    # L2 strength for the mitigation learner
DORMANT_THRESHOLD = 0.0  # a ReLU unit is "dormant" if it is >= this share inactive
DORMANT_INACTIVE_FRAC = 0.99  # inactive on >= 99% of probe samples => dormant


class MLP(nn.Module):
    """Two-hidden-layer ReLU MLP. Exposes hidden activations for the dormant probe."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x, return_hidden=False):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        y = self.out(h2)
        if return_hidden:
            return y, (h1, h2)
        return y


def make_teacher(in_dim, gen):
    """A random WIDE two-layer nonlinear teacher (TEACHER_HIDDEN units). A fresh
    teacher is drawn per task so the target mapping keeps changing (the overwriting
    non-stationarity the survey uses to induce plasticity loss). The teacher is
    wider than the student, so the student cannot perfectly fit it in the tight
    per-task budget."""
    w1 = torch.tensor(gen.standard_normal((in_dim, TEACHER_HIDDEN)) / np.sqrt(in_dim),
                      dtype=torch.float32)
    w2 = torch.tensor(gen.standard_normal((TEACHER_HIDDEN, 1)) / np.sqrt(TEACHER_HIDDEN),
                      dtype=torch.float32)
    return w1, w2


def make_task(base_inputs, teacher):
    """Build one task: fixed inputs, labels from a fresh random nonlinear teacher."""
    w1, w2 = teacher
    h = torch.relu(base_inputs @ w1)
    logits = h @ w2
    y = (logits > logits.median()).float()  # balanced binary labels
    return base_inputs, y


def dormant_fraction(model, probe_x):
    """Fraction of hidden ReLU units that are inactive on ~all probe samples."""
    model.eval()
    with torch.no_grad():
        _, (h1, h2) = model(probe_x, return_hidden=True)
    acts = torch.cat([h1, h2], dim=1)          # (n_probe, 2*HIDDEN)
    active = (acts > DORMANT_THRESHOLD).float()  # 1 when unit fired
    inactive_frac = 1.0 - active.mean(dim=0)     # per-unit share of samples inactive
    dormant = (inactive_frac >= DORMANT_INACTIVE_FRAC).float().mean().item()
    return dormant


def run_condition(name, weight_decay, base_inputs, task_teachers, probe_x):
    """Train ONE learner across the whole task stream; record per-task metrics.
    task_teachers is a list of per-task teachers, shared across conditions so both
    learners see an identical task stream."""
    torch.manual_seed(SEED)  # identical init for both conditions
    model = MLP(INPUT_DIM, HIDDEN)
    opt = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    rows = []
    for t in range(N_TASKS):
        x, y = make_task(base_inputs, task_teachers[t])
        model.train()
        best_loss = float("inf")
        for _ in range(STEPS_PER_TASK):
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            best_loss = min(best_loss, float(loss.item()))
        # new-task metrics AFTER the fixed budget = plasticity signal
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) > 0.5).float()
            acc = (pred == y).float().mean().item()
        dfrac = dormant_fraction(model, probe_x)
        rows.append(
            {"condition": name, "task": t, "new_task_accuracy": acc,
             "dormant_fraction": dfrac, "best_loss": best_loss,
             "final_loss": float(loss.item())}
        )
    return rows


def main():
    gen = np.random.default_rng(SEED)
    base_inputs = torch.tensor(
        gen.standard_normal((N_SAMPLES, INPUT_DIM)), dtype=torch.float32
    )
    probe_x = torch.tensor(
        gen.standard_normal((256, INPUT_DIM)), dtype=torch.float32
    )
    teacher = None  # unused: a fresh teacher is drawn per task below
    task_teachers = [make_teacher(INPUT_DIM, gen) for _ in range(N_TASKS)]

    all_rows = []
    all_rows += run_condition("plain_sgd", 0.0, base_inputs, task_teachers, probe_x)
    all_rows += run_condition("sgd_weight_decay", WEIGHT_DECAY, base_inputs,
                              task_teachers, probe_x)

    # ---- write per-task CSV ----
    csv_path = os.path.join(ANALYSIS, "plasticity_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("condition,task,new_task_accuracy,dormant_fraction,best_loss,final_loss\n")
        for r in all_rows:
            f.write(f"{r['condition']},{r['task']},{r['new_task_accuracy']:.4f},"
                    f"{r['dormant_fraction']:.4f},{r['best_loss']:.4f},"
                    f"{r['final_loss']:.4f}\n")

    # ---- headline summary (early vs late window means) ----
    def window(cond, key, lo, hi):
        vals = [r[key] for r in all_rows if r["condition"] == cond
                and lo <= r["task"] < hi]
        return float(np.mean(vals))

    early_lo, early_hi = 0, 5
    late_lo, late_hi = N_TASKS - 5, N_TASKS

    def block(cond):
        d = {
            "acc_first5": round(window(cond, "new_task_accuracy", early_lo, early_hi), 4),
            "acc_last5": round(window(cond, "new_task_accuracy", late_lo, late_hi), 4),
            "loss_first5": round(window(cond, "best_loss", early_lo, early_hi), 4),
            "loss_last5": round(window(cond, "best_loss", late_lo, late_hi), 4),
            "dormant_first5": round(window(cond, "dormant_fraction", early_lo, early_hi), 4),
            "dormant_last5": round(window(cond, "dormant_fraction", late_lo, late_hi), 4),
        }
        d["acc_drop"] = round(d["acc_first5"] - d["acc_last5"], 4)
        d["loss_rise"] = round(d["loss_last5"] - d["loss_first5"], 4)
        d["dormant_rise"] = round(d["dormant_last5"] - d["dormant_first5"], 4)
        return d

    summary = {
        "seed": SEED, "n_tasks": N_TASKS, "input_dim": INPUT_DIM,
        "student_hidden": HIDDEN, "teacher_hidden": TEACHER_HIDDEN,
        "steps_per_task": STEPS_PER_TASK, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "plain_sgd": block("plain_sgd"),
        "sgd_weight_decay": block("sgd_weight_decay"),
    }
    with open(os.path.join(ANALYSIS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(cond, key):
        return [r[key] for r in all_rows if r["condition"] == cond]

    tasks = list(range(N_TASKS))

    plt.figure(figsize=(7, 4.2))
    plt.plot(tasks, series("plain_sgd", "best_loss"), marker="o", ms=3,
             label="Plain SGD")
    plt.plot(tasks, series("sgd_weight_decay", "best_loss"), marker="s", ms=3,
             label="SGD + weight decay")
    plt.xlabel("Task index (non-stationary stream)")
    plt.ylabel("Best training loss reached (fixed budget)")
    plt.title("Plasticity: best fit reachable on each NEW task over the stream")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "learning_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.2))
    plt.plot(tasks, series("plain_sgd", "dormant_fraction"), marker="o", ms=3,
             label="Plain SGD")
    plt.plot(tasks, series("sgd_weight_decay", "dormant_fraction"), marker="s", ms=3,
             label="SGD + weight decay")
    plt.xlabel("Task index (non-stationary stream)")
    plt.ylabel("Dormant ReLU-unit fraction")
    plt.title("Mechanism: dormant (dead) units over the stream")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "dormant_curve.png"), dpi=150)
    plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
