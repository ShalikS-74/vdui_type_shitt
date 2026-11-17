from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 0
WARNING_CUTOFFS = [0.5, 0.7]
BASE_FEATURES = ["Pop", "Cap", "Hier", "Gov", "Infra", "Info", "Money"]
FULL_FEATURES = BASE_FEATURES + [f"{feature}_delta" for feature in BASE_FEATURES]
C_GRID = [0.01, 0.1, 1.0, 10.0]


def make_model(features: list[str], c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=c_value, class_weight="balanced", max_iter=1000, random_state=SEED
                ),
            ),
        ]
    )


def choose_c(x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> float:
    candidates = []
    splitter = GroupKFold(n_splits=min(3, groups.nunique()))
    for c_value in C_GRID:
        scores = []
        for train_index, validation_index in splitter.split(x, y, groups):
            if y.iloc[train_index].nunique() < 2 or y.iloc[validation_index].nunique() < 2:
                continue
            fitted = make_model(list(x.columns), c_value).fit(
                x.iloc[train_index], y.iloc[train_index]
            )
            probabilities = fitted.predict_proba(x.iloc[validation_index])[:, 1]
            scores.append(average_precision_score(y.iloc[validation_index], probabilities))
        candidates.append((np.mean(scores) if scores else -np.inf, c_value))
    return max(candidates, key=lambda item: (item[0], -item[1]))[1]


def evaluate_variant(table: pd.DataFrame, features: list[str], tuned: bool) -> pd.DataFrame:
    x = table[features]
    y = table["contraction"].astype(int)
    groups = table["polity"]
    probabilities = np.full(len(table), np.nan)
    selected_c = []
    outer = LeaveOneGroupOut()
    for train_index, test_index in outer.split(x, y, groups):
        c_value = choose_c(x.iloc[train_index], y.iloc[train_index], groups.iloc[train_index]) if tuned else 1.0
        if y.iloc[train_index].nunique() < 2:
            continue
        fitted = make_model(features, c_value).fit(x.iloc[train_index], y.iloc[train_index])
        probabilities[test_index] = fitted.predict_proba(x.iloc[test_index])[:, 1]
        selected_c.append(c_value)
    result = table[["polity", "century", "contraction"]].copy()
    result["p"] = probabilities
    result["variant"] = ("seven_inputs" if features == BASE_FEATURES else "full_inputs") + ("_tuned" if tuned else "_fixed_c1")
    result["selected_c_summary"] = ",".join(map(str, sorted(set(selected_c))))
    return result


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in predictions.groupby("variant"):
        valid = group["p"].notna()
        y = group.loc[valid, "contraction"].astype(int)
        probabilities = group.loc[valid, "p"]
        row = {"variant": variant, "rows": len(y), "positive_cases": int(y.sum()), "pr_auc": average_precision_score(y, probabilities)}
        for cutoff in WARNING_CUTOFFS:
            warnings = probabilities >= cutoff
            row[f"warnings_at_{cutoff}"] = int(warnings.sum())
            row[f"real_cases_caught_at_{cutoff}"] = int(y[ warnings ].sum())
            row[f"false_warnings_at_{cutoff}"] = int((warnings & y.eq(0)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    table = pd.read_parquet(root / "data" / "table.parquet").reset_index(drop=True)
    assert table.duplicated(["polity", "century"]).sum() == 0
    assert table["contraction"].value_counts().to_dict() == {0: 92, 1: 6}
    audit = pd.read_csv(root / "data" / "historical_case_audit.csv", keep_default_na=False)
    assert audit["verified_correction"].eq("none").all()

    predictions = pd.concat(
        [
            evaluate_variant(table, FULL_FEATURES, tuned=False),
            evaluate_variant(table, BASE_FEATURES, tuned=False),
            evaluate_variant(table, FULL_FEATURES, tuned=True),
            evaluate_variant(table, BASE_FEATURES, tuned=True),
        ],
        ignore_index=True,
    )
    summary = summarize(predictions)
    predictions.to_csv(root / "data" / "experiments" / "model_variant_predictions.csv", index=False)
    summary.to_csv(root / "data" / "experiments" / "model_variant_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved: {root / 'data/experiments/model_variant_predictions.csv'}")
    print(f"saved: {root / 'data/experiments/model_variant_summary.csv'}")
    print("outer evaluation: LeaveOneGroupOut by composite polity")
    print("tuned variants: C selected by grouped validation inside each outer training set only")
    print("data check: 92 non-loss cases retained; no duplicate polity-century records")
    print("historical corrections: none verified, so labels were unchanged")


if __name__ == "__main__":
    main()