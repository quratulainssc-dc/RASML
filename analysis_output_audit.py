#!/usr/bin/env python3
"""Audit inputs and the numerical outputs produced by the locked pipeline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "revision_20260827"
OUT = ROOT / "final_locked_results"

def main():
    problems = []
    def require(condition, message):
        if not condition:
            problems.append(message)

    data = pd.read_csv(DATA / "analysis_matrix.csv")
    folds = pd.read_csv(DATA / "fixed_folds.csv")
    scores = pd.read_csv(OUT / "01_repeated_oof_scores.csv")
    performance = pd.read_csv(OUT / "07_primary_consensus_performance.csv")
    manifest = json.loads((OUT / "19_frozen_manifest.json").read_text())

    features = data.drop(columns=["Author Name", "Class"])
    require(data.shape == (1184, 66), "Unexpected analytical-matrix dimensions")
    require(int(data["Class"].sum()) == 609, "Unexpected awardee count")
    require(int((data["Class"] == 0).sum()) == 575, "Unexpected comparison count")
    require(features.shape[1] == 64, "Expected 64 numerical indicators")
    require(np.isfinite(features.to_numpy()).all(), "Non-finite indicator value detected")
    require(not features.isna().any().any(), "Missing indicator value detected")
    require(folds["identity_group"].nunique() == 1154, "Unexpected identity-group count")
    require(not folds.groupby("identity_group")["outer_fold"].nunique().gt(1).any(),
            "An identity group crosses archived outer folds")
    require(len(scores) == 10 * 1184 * 7, "Unexpected repeated-score row count")
    require(scores.groupby(["repeat", "record", "method"]).size().eq(1).all(),
            "Repeated scores are not unique by repeat, record, and method")
    require(np.isfinite(scores[["raw_score", "normalized_score"]]).all().all(),
            "Non-finite held-out score detected")
    require(set(performance["method"]) ==
            {"LR", "RF", "HGB", "XGB", "Composite", "Panel5", "HGB5"},
            "A reported method is missing from primary performance")
    require(all(len(grid) == 12 for grid in manifest["grids"].values()),
            "Model search grids do not all contain 12 configurations")
    require(manifest["class_weighting"] == "none for every model",
            "Unexpected class-weighting specification")

    print("ANALYSIS AUDIT")
    print("Status:", "PASS" if not problems else "FAIL")
    print("Issues:", len(problems))
    for problem in problems:
        print("-", problem)
    if problems:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
