from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 0
HORIZON = 100
LOSS_RULES = {"20pct": np.log10(0.8), "30pct": np.log10(0.7), "50pct": np.log10(0.5)}
FEATURES = ["Pop", "Cap", "Hier", "Gov", "Infra", "Info", "Money"]
FEATURES += [f"{feature}_delta" for feature in FEATURES]


def model() -> Pipeline:
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]
    )


def build_labels(root: Path) -> pd.DataFrame:
    raw = pd.read_excel(root / "data/raw/Equinox_on_GitHub_June9_2022.xlsx", sheet_name="AggrSCWarAgriRelig")
    raw = raw.replace(-999, np.nan)
    raw = raw[raw["uniq"].astype(str).str.lower().ne("n")].copy()
    raw["Time"] = pd.to_numeric(raw["Time"], errors="coerce")
    raw["Terr"] = pd.to_numeric(raw["Terr"], errors="coerce")
    current = raw[["NGA", "PolID", "Time", "Terr"] + FEATURES[:7]].copy()
    future = current[["NGA", "PolID", "Time", "Terr"]].rename(columns={"Time": "next_time", "Terr": "next_Terr"})
    data = current.assign(next_time=current["Time"] + HORIZON).merge(
        future, on=["NGA", "PolID", "next_time"], how="left"
    )
    previous = current[["NGA", "PolID", "Time"] + FEATURES[:7]].copy()
    previous["Time"] += HORIZON
    previous = previous.rename(columns={feature: f"{feature}_prev" for feature in FEATURES[:7]})
    data = data.merge(previous, on=["NGA", "PolID", "Time"], how="left")
    for feature in FEATURES[:7]:
        data[f"{feature}_delta"] = data[feature] - data[f"{feature}_prev"]
    data = data.dropna(subset=["Terr", "next_Terr"]).copy()
    data["polity"] = data["NGA"] + "::" + data["PolID"]
    return data


def evaluate(x: pd.DataFrame, y: pd.Series, groups: pd.Series, splitter) -> tuple[float, float, float]:
    probabilities = np.full(len(y), np.nan)
    for train_index, test_index in splitter.split(x, y, groups):
        if y.iloc[train_index].nunique() < 2:
            continue
        fitted = model().fit(x.iloc[train_index], y.iloc[train_index])
        probabilities[test_index] = fitted.predict_proba(x.iloc[test_index])[:, 1]
    valid = np.isfinite(probabilities)
    if y.iloc[valid].nunique() < 2:
        return (np.nan, np.nan, np.nan)
    predicted = probabilities[valid] >= 0.5
    return (
        precision_score(y.iloc[valid], predicted, zero_division=0),
        recall_score(y.iloc[valid], predicted, zero_division=0),
        average_precision_score(y.iloc[valid], probabilities[valid]),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = build_labels(root)
    grouped = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    loo = LeaveOneGroupOut()
    rows = []
    for name, threshold in LOSS_RULES.items():
        labeled = data[data["Terr"].notna() & data["next_Terr"].notna()].copy()
        labeled["label"] = (labeled["next_Terr"] - labeled["Terr"] <= threshold).astype(int)
        y = labeled["label"]
        x = labeled[FEATURES]
        groups = labeled["polity"]
        precision, recall, pr_auc = evaluate(x, y, groups, grouped)
        loo_precision, loo_recall, loo_pr_auc = evaluate(x, y, groups, loo)
        rows.append(
            {
                "rule": name,
                "threshold_log10_change": threshold,
                "rows": len(labeled),
                "positive_cases": int(y.sum()),
                "grouped_precision": precision,
                "grouped_recall": recall,
                "grouped_pr_auc": pr_auc,
                "leave_polity_out_precision": loo_precision,
                "leave_polity_out_recall": loo_recall,
                "leave_polity_out_pr_auc": loo_pr_auc,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(root / "data" / "label_sensitivity.csv", index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved: {root / 'data/label_sensitivity.csv'}")
    print("CrisisDB comparison: not run; no CrisisDB file is present in data/raw.")
    print("Interpretation: leave-polity-out tests states/groups not used for fitting each prediction.")
    print("Caution: these estimates remain unstable because the positive counts are small.")


if __name__ == "__main__":
    main()