"""Experiment register (rule R3).

One row per experiment ID, stored in results/experiment_register.csv.
Re-running an experiment overwrites its own row instead of duplicating it.
"""
import pandas as pd

from .data import ROOT

REGISTER_FILE = ROOT / "results" / "experiment_register.csv"
COLUMNS = ["id", "task", "question", "hypothesis", "changed_variable", "fixed_conditions",
           "split", "seed", "configuration", "metric", "result", "output_path"]


def log_experiment(**row):
    missing = set(COLUMNS) - set(row)
    if missing:
        raise ValueError(f"register row is missing: {sorted(missing)}")
    if REGISTER_FILE.exists():
        df = pd.read_csv(REGISTER_FILE, dtype=str)
        df = df[df["id"] != row["id"]]
    else:
        df = pd.DataFrame(columns=COLUMNS)
    df = pd.concat([df, pd.DataFrame([{k: str(row[k]) for k in COLUMNS}])], ignore_index=True)
    df = df.sort_values("id", kind="stable")
    REGISTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REGISTER_FILE, index=False)
    return df


def show_register():
    return pd.read_csv(REGISTER_FILE) if REGISTER_FILE.exists() else pd.DataFrame(columns=COLUMNS)
