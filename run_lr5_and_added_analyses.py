#!/usr/bin/env python3
"""Run the added LR5 analysis and, when archived consensus scores are present,
the added paired-inference and ordering-comparison analyses.

The implementation preserves the locked study design: the archived outer-fold
assignment for repeat 1, deterministically regenerated grouped folds for repeats
2--10, training-only five-indicator selection, three-fold grouped inner
validation, the frozen 12-value LR grid, within-outer-fold percentile ranks,
and consensus scores averaged across ten repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


MASTER_SEED = 20260819
N_REPEATS = 10
N_OUTER = 5
N_INNER = 3
N_BOOTSTRAP = 2000
N_RANDOMIZATION = 10000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs(data: pd.DataFrame, fixed: pd.DataFrame) -> None:
    if data.shape != (1184, 66):
        raise ValueError(f"Expected analytical matrix (1184, 66), found {data.shape}")
    if int(data["Class"].sum()) != 609:
        raise ValueError("Expected 609 awardee labels")
    if fixed.shape[0] != len(data):
        raise ValueError("Fold file and analytical matrix have different row counts")
    if not data["Author Name"].astype(str).equals(fixed["Author Name"].astype(str)):
        raise ValueError("Analytical matrix and fold file are not in identical row order")
    if not data["Class"].astype(int).equals(fixed["Class"].astype(int)):
        raise ValueError("Class labels disagree between analytical matrix and fold file")
    if fixed["identity_group"].nunique() != 1154:
        raise ValueError("Expected 1,154 identity groups")
    if fixed.groupby("identity_group")["outer_fold"].nunique().gt(1).any():
        raise ValueError("An identity group crosses archived outer folds")
    feature_values = data.drop(columns=["Author Name", "Class"]).to_numpy(float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite feature values detected")


def outer_assignments(data: pd.DataFrame, fixed: pd.DataFrame) -> list[np.ndarray]:
    labels = data["Class"].to_numpy(int)
    groups = fixed["identity_group"].to_numpy()
    assignments = [fixed["outer_fold"].to_numpy(int)]
    for repeat_offset in range(1, 10):
        fold = np.zeros(len(labels), dtype=int)
        splitter = StratifiedGroupKFold(
            n_splits=N_OUTER,
            shuffle=True,
            random_state=MASTER_SEED + repeat_offset,
        )
        for fold_number, (_, test_indices) in enumerate(
            splitter.split(data, labels, groups), start=1
        ):
            fold[test_indices] = fold_number
        assignments.append(fold)
    return assignments


def top_metrics(labels: np.ndarray, scores: np.ndarray, proportion: float = 0.10):
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    cutoff = math.ceil(len(labels) * proportion)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    counts = np.diff(np.r_[starts, len(labels)])
    relevance = np.repeat(np.add.reduceat(labels[order], starts) / counts, counts)[:cutoff]
    discounts = 1 / np.log2(np.arange(2, cutoff + 2))
    ideal = discounts[: min(cutoff, int(labels.sum()))].sum()
    return {
        "NDCG10": float(relevance @ discounts / ideal),
        "Precision10": float(relevance.mean()),
        "Recall10": float(relevance.sum() / labels.sum()),
        "k": int(cutoff),
    }


def rank_indicators(features: pd.DataFrame, labels: np.ndarray) -> list[str]:
    rows = []
    for column in features.columns:
        result = top_metrics(labels, features[column].to_numpy(float))
        rows.append((column, result["NDCG10"], result["Precision10"]))
    return [
        row[0]
        for row in sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))
    ]


def select5(features: pd.DataFrame, labels: np.ndarray) -> list[str]:
    order = rank_indicators(features, labels)
    correlations = features.corr(method="spearman").abs()
    selected = []
    for column in order:
        if all(correlations.loc[column, existing] < 0.90 for existing in selected):
            selected.append(column)
        if len(selected) == 5:
            break
    if len(selected) != 5:
        raise RuntimeError("Five nonredundant indicators could not be selected")
    return selected


def all_metrics(labels: np.ndarray, scores: np.ndarray):
    from sklearn.metrics import average_precision_score, roc_auc_score

    result = top_metrics(labels, scores)
    result.update(
        {
            "ROC-AUC": float(roc_auc_score(labels, scores)),
            "AP": float(average_precision_score(labels, scores)),
        }
    )
    return result


def fold_percentile(scores: np.ndarray, fold: np.ndarray) -> np.ndarray:
    normalized = np.empty(len(scores), dtype=float)
    for fold_number in range(1, N_OUTER + 1):
        indices = np.flatnonzero(fold == fold_number)
        normalized[indices] = (
            rankdata(scores[indices], method="average") - 1
        ) / (len(indices) - 1)
    return normalized


def fit_lr(
    parameters: dict,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    test_features: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train_array = imputer.fit_transform(train_features)
    test_array = imputer.transform(test_features)
    scaler = StandardScaler()
    train_array = scaler.fit_transform(train_array)
    test_array = scaler.transform(test_array)
    model = LogisticRegression(
        C=float(parameters["C"]),
        max_iter=5000,
        solver="lbfgs",
        class_weight=None,
        random_state=seed,
    )
    model.fit(train_array, train_labels)
    return model.predict_proba(test_array)[:, 1]


def inner_candidate_score(
    parameters: dict,
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    splits,
    seed: int,
) -> tuple[float, list[str]]:
    fold_scores = []
    selections = []
    for inner_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        selected = select5(features.iloc[train_indices], labels[train_indices])
        predictions = fit_lr(
            parameters,
            features.iloc[train_indices][selected],
            labels[train_indices],
            features.iloc[validation_indices][selected],
            seed + inner_number,
        )
        fold_scores.append(
            top_metrics(labels[validation_indices], predictions)["NDCG10"]
        )
        selections.append(" | ".join(selected))
    return float(np.mean(fold_scores)), selections


def run_lr5(
    data: pd.DataFrame,
    fixed: pd.DataFrame,
    lr_grid: list[dict],
    output_dir: Path,
    jobs: int,
):
    features = data.drop(columns=["Author Name", "Class"])
    labels = data["Class"].to_numpy(int)
    groups = fixed["identity_group"].to_numpy()
    repeats = outer_assignments(data, fixed)

    score_rows = []
    repeat_rows = []
    fold_rows = []
    configuration_rows = []
    inner_rows = []

    for repeat_number, fold in enumerate(repeats, start=1):
        raw_scores = np.full(len(labels), np.nan)
        for fold_number in range(1, N_OUTER + 1):
            train_indices = np.flatnonzero(fold != fold_number)
            test_indices = np.flatnonzero(fold == fold_number)
            if set(groups[train_indices]) & set(groups[test_indices]):
                raise RuntimeError("Identity leakage detected")

            seed = MASTER_SEED + repeat_number * 100 + fold_number
            inner_splits = list(
                StratifiedGroupKFold(
                    n_splits=N_INNER,
                    shuffle=True,
                    random_state=seed,
                ).split(
                    features.iloc[train_indices],
                    labels[train_indices],
                    groups[train_indices],
                )
            )

            evaluations = Parallel(n_jobs=jobs)(
                delayed(inner_candidate_score)(
                    parameters,
                    features.iloc[train_indices],
                    labels[train_indices],
                    groups[train_indices],
                    inner_splits,
                    seed * 100 + candidate_index * 10,
                )
                for candidate_index, parameters in enumerate(lr_grid)
            )
            mean_scores = [evaluation[0] for evaluation in evaluations]
            selected_index = sorted(
                range(len(lr_grid)), key=lambda index: (-mean_scores[index], index)
            )[0]
            selected_features = select5(
                features.iloc[train_indices], labels[train_indices]
            )
            predictions = fit_lr(
                lr_grid[selected_index],
                features.iloc[train_indices][selected_features],
                labels[train_indices],
                features.iloc[test_indices][selected_features],
                seed,
            )
            raw_scores[test_indices] = predictions

            configuration_rows.append(
                {
                    "repeat": repeat_number,
                    "fold": fold_number,
                    "candidate": selected_index + 1,
                    "C": lr_grid[selected_index]["C"],
                    "inner_NDCG10": mean_scores[selected_index],
                    "features": " | ".join(selected_features),
                }
            )
            for candidate_index, (parameters, evaluation) in enumerate(
                zip(lr_grid, evaluations), start=1
            ):
                inner_rows.append(
                    {
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "candidate": candidate_index,
                        "C": parameters["C"],
                        "mean_inner_NDCG10": evaluation[0],
                        "inner_selected_features": " || ".join(evaluation[1]),
                    }
                )

            fold_result = all_metrics(labels[test_indices], predictions)
            fold_rows.append(
                {
                    "repeat": repeat_number,
                    "fold": fold_number,
                    **fold_result,
                }
            )
            print(
                f"LR5 repeat {repeat_number}/{N_REPEATS}, "
                f"fold {fold_number}/{N_OUTER} complete",
                flush=True,
            )

        if not np.isfinite(raw_scores).all():
            raise RuntimeError("Incomplete held-out LR5 scores")
        normalized_scores = fold_percentile(raw_scores, fold)
        repeat_result = all_metrics(labels, normalized_scores)
        repeat_rows.append({"repeat": repeat_number, **repeat_result})
        score_rows.extend(
            {
                "repeat": repeat_number,
                "record": record,
                "fold": int(fold[record]),
                "raw_score": raw_scores[record],
                "normalized_score": normalized_scores[record],
            }
            for record in range(len(labels))
        )

    scores = pd.DataFrame(score_rows)
    repeat_performance = pd.DataFrame(repeat_rows)
    fold_performance = pd.DataFrame(fold_rows)
    configurations = pd.DataFrame(configuration_rows)
    inner_results = pd.DataFrame(inner_rows)

    consensus = (
        scores.pivot(index="record", columns="repeat", values="normalized_score")
        .sort_index()
        .mean(axis=1)
        .rename("LR5")
        .reset_index()
    )
    consensus.insert(1, "Class", labels)
    consensus_metrics = pd.DataFrame([{"method": "LR5", **all_metrics(labels, consensus["LR5"])}])
    summary = pd.DataFrame(
        [
            {
                "method": "LR5",
                **{
                    f"{metric}_{statistic}": getattr(repeat_performance[metric], statistic)()
                    for metric in ["NDCG10", "Precision10", "Recall10", "ROC-AUC", "AP"]
                    for statistic in ["mean", "std", "min", "max"]
                },
            }
        ]
    )

    scores.to_csv(output_dir / "lr5_repeated_oof_scores.csv", index=False)
    repeat_performance.to_csv(output_dir / "lr5_repeat_performance.csv", index=False)
    fold_performance.to_csv(output_dir / "lr5_all_fold_performance.csv", index=False)
    fold_performance.query("repeat == 1").to_csv(
        output_dir / "lr5_archived_repeat1_fold_values.csv", index=False
    )
    configurations.to_csv(output_dir / "lr5_selected_configurations.csv", index=False)
    inner_results.to_csv(output_dir / "lr5_inner_search_results.csv", index=False)
    consensus.to_csv(output_dir / "lr5_consensus_scores.csv", index=False)
    consensus_metrics.to_csv(output_dir / "lr5_consensus_performance.csv", index=False)
    summary.to_csv(output_dir / "lr5_repeat_summary.csv", index=False)
    return labels, groups, consensus, consensus_metrics, summary


def resample_structure(labels: np.ndarray, groups: np.ndarray):
    group_values, inverse = np.unique(groups, return_inverse=True)
    records_by_group = [np.flatnonzero(inverse == index) for index in range(len(group_values))]
    strata = {}
    for group_index, record_indices in enumerate(records_by_group):
        key = (len(record_indices), int(labels[record_indices[0]]))
        strata.setdefault(key, []).append(group_index)
    return group_values, inverse, records_by_group, strata


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    order = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p), dtype=float)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(raw_p[order] * (len(raw_p) - np.arange(len(raw_p)))),
    )
    return adjusted


def added_inference(
    labels: np.ndarray,
    groups: np.ndarray,
    scores: pd.DataFrame,
    output_dir: Path,
):
    contrasts = [
        ("HGB5", "Panel5"),
        ("LR5", "Panel5"),
        ("LR5", "HGB5"),
    ]
    group_values, inverse, records_by_group, strata = resample_structure(labels, groups)
    bootstrap_rng = np.random.default_rng(MASTER_SEED + 13000)
    randomization_rng = np.random.default_rng(MASTER_SEED + 13001)
    rows = []
    for method_a, method_b in contrasts:
        first = scores[method_a].to_numpy(float)
        second = scores[method_b].to_numpy(float)
        observed = top_metrics(labels, first)["NDCG10"] - top_metrics(labels, second)["NDCG10"]
        bootstrap_differences = []
        for _ in range(N_BOOTSTRAP):
            sampled_indices = np.concatenate(
                [
                    records_by_group[group_index]
                    for stratum_groups in strata.values()
                    for group_index in bootstrap_rng.choice(
                        stratum_groups, len(stratum_groups), replace=True
                    )
                ]
            )
            bootstrap_differences.append(
                top_metrics(labels[sampled_indices], first[sampled_indices])["NDCG10"]
                - top_metrics(labels[sampled_indices], second[sampled_indices])["NDCG10"]
            )
        extreme = 0
        for _ in range(N_RANDOMIZATION):
            swap = (randomization_rng.random(len(group_values)) < 0.5)[inverse]
            permuted_first = np.where(swap, second, first)
            permuted_second = np.where(swap, first, second)
            difference = (
                top_metrics(labels, permuted_first)["NDCG10"]
                - top_metrics(labels, permuted_second)["NDCG10"]
            )
            extreme += abs(difference) >= abs(observed) - 1e-15
        rows.append(
            {
                "analysis_set": "added contrasts",
                "method_a": method_a,
                "method_b": method_b,
                "difference": observed,
                "ci_low": np.quantile(bootstrap_differences, 0.025),
                "ci_high": np.quantile(bootstrap_differences, 0.975),
                "raw_p": (extreme + 1) / (N_RANDOMIZATION + 1),
                "bootstrap_replicates": N_BOOTSTRAP,
                "randomization_draws": N_RANDOMIZATION,
            }
        )
    results = pd.DataFrame(rows)
    results["holm_p"] = holm_adjust(results["raw_p"].to_numpy())
    results.to_csv(output_dir / "added_three_contrasts.csv", index=False)
    return results


def deterministic_top_set(scores: np.ndarray, count: int = 119) -> set[int]:
    # Stable sorting matches the locked top-k implementation and makes membership
    # deterministic when percentile averaging creates exact ties.
    return set(np.argsort(-np.asarray(scores), kind="stable")[:count].tolist())


def ordering_comparisons(labels: np.ndarray, scores: pd.DataFrame, output_dir: Path):
    comparisons = [
        ("HGB5", "Panel5", "HGB5", "Panel5"),
        ("LR", "HGB", "LR", "HGB64"),
        ("Composite", "HGB", "Composite", "HGB64"),
    ]
    rows = []
    for source_a, source_b, method_a, method_b in comparisons:
        first = scores[source_a].to_numpy(float)
        second = scores[source_b].to_numpy(float)
        top_a = deterministic_top_set(first)
        top_b = deterministic_top_set(second)
        shared = top_a & top_b
        only_a = top_a - top_b
        only_b = top_b - top_a
        rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "spearman_rho": spearmanr(first, second).statistic,
                "kendall_tau_b": kendalltau(first, second, variant="b").statistic,
                "top_count_each": 119,
                "shared_top119": len(shared),
                "jaccard": len(shared) / len(top_a | top_b),
                "awardees_unique_to_a": int(labels[list(only_a)].sum()),
                "awardees_unique_to_b": int(labels[list(only_b)].sum()),
                "total_unique_to_a": len(only_a),
                "total_unique_to_b": len(only_b),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "ordering_and_top119_overlap.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = arguments.input_dir / "analytical_matrix.xlsx"
    folds_path = arguments.input_dir / "fixed_folds.csv"
    manifest_path = arguments.input_dir / "19_frozen_manifest.json"
    pipeline_path = arguments.input_dir / "final_locked_analysis_pipeline.py"
    archived_scores_path = arguments.input_dir / "08_primary_consensus_scores.csv"

    data = pd.read_excel(dataset_path, sheet_name="Updated 64 indicators")
    fixed = pd.read_csv(folds_path)
    validate_inputs(data, fixed)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["master_seed"] != MASTER_SEED:
        raise ValueError("Unexpected master seed")
    lr_grid = manifest["grids"]["LR"]
    if len(lr_grid) != 12:
        raise ValueError("Frozen manifest does not contain 12 LR configurations")
    labels, groups, lr5_scores, _, _ = run_lr5(
        data, fixed, lr_grid, arguments.output_dir, arguments.jobs
    )

    audit = {
        "dataset_sha256": sha256(dataset_path),
        "folds_sha256": sha256(folds_path),
        "manifest_sha256": sha256(manifest_path),
        "pipeline_sha256": sha256(pipeline_path),
        "n_records": len(data),
        "n_awardees": int(labels.sum()),
        "n_identity_groups": int(pd.Series(groups).nunique()),
        "lr_candidates": len(lr_grid),
        "outer_repeats": N_REPEATS,
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
    }

    if archived_scores_path.exists():
        archived = pd.read_csv(archived_scores_path).sort_values("record").reset_index(drop=True)
        needed = {"record", "Class", "LR", "HGB", "Composite", "Panel5", "HGB5"}
        missing = sorted(needed - set(archived.columns))
        if missing:
            raise ValueError(f"Archived scores are missing columns: {missing}")
        if not archived["record"].to_numpy(int).tolist() == list(range(len(labels))):
            raise ValueError("Archived consensus scores are not in complete record order")
        if not np.array_equal(archived["Class"].to_numpy(int), labels):
            raise ValueError("Archived consensus labels disagree with the analytical matrix")
        combined = archived.merge(lr5_scores[["record", "LR5"]], on="record", validate="one_to_one")
        added_inference(labels, groups, combined, arguments.output_dir)
        ordering_comparisons(labels, combined, arguments.output_dir)
        combined.to_csv(arguments.output_dir / "consensus_scores_with_lr5.csv", index=False)
        audit["archived_score_analyses_completed"] = True
    else:
        audit["archived_score_analyses_completed"] = False
        audit["missing_required_file"] = "08_primary_consensus_scores.csv"

    (arguments.output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
