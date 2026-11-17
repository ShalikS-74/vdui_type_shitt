from pathlib import Path

import pandas as pd


CASES = [
    {
        "case_type": "false_alarm",
        "polity": "Latium::ItRomPr",
        "time_t": 0,
        "source": "https://www.britannica.com/place/Roman-Empire",
        "note": "Equinox Terr rises from 6.653213 to 6.763428. Roman power was consolidating and expanding in the Principate; no verified territory-loss correction.",
    },
    {
        "case_type": "false_alarm",
        "polity": "Latium::ItRomPr",
        "time_t": 100,
        "source": "https://www.britannica.com/place/Roman-Empire",
        "note": "Equinox Terr falls slightly from 6.763428 to 6.698970, below the 30% log10 threshold. This is a modest modeled change, not a verified major loss.",
    },
    {
        "case_type": "false_alarm",
        "polity": "Middle Yellow River Valley::CnMing*",
        "time_t": 1400,
        "source": "https://www.britannica.com/topic/Ming-dynasty-Chinese-history",
        "note": "Equinox Terr rises from 6.591065 to 6.658011. Ming rule continued through this window; no verified correction.",
    },
    {
        "case_type": "false_alarm",
        "polity": "Middle Yellow River Valley::CnQingL",
        "time_t": 1800,
        "source": "https://www.britannica.com/place/Qing-dynasty",
        "note": "Equinox Terr is unchanged at 7.053078. Qing losses and foreign concessions occurred later in the 19th century, but do not justify changing this 1800-1900 label.",
    },
    {
        "case_type": "false_alarm",
        "polity": "Middle Yellow River Valley::CnWHan*",
        "time_t": -200,
        "source": "https://www.britannica.com/place/Han-dynasty",
        "note": "Equinox Terr rises from 6.386142 to 6.659631. The Western Han remained an established imperial regime; no verified correction.",
    },
    {
        "case_type": "contraction",
        "polity": "Latium::ItPapEM",
        "time_t": 1000,
        "source": "https://www.britannica.com/place/Papal-States",
        "note": "Equinox Terr falls from 4.146128 to 3.778151. Papal territory and control varied, but the source confirms medieval territorial fluctuations; retain as an Equinox land-loss case pending a quantitative historical map check.",
    },
    {
        "case_type": "contraction",
        "polity": "Middle Yellow River Valley::CnEHan*",
        "time_t": 100,
        "source": "https://www.britannica.com/place/Han-dynasty",
        "note": "Equinox Terr falls from 6.812913 to 6.593508. Han authority continued through this period, so the label is territorial contraction, not regime collapse; no verified data correction.",
    },
    {
        "case_type": "contraction",
        "polity": "Southern Mesopotamia::IqAbbs1",
        "time_t": 800,
        "source": "https://www.britannica.com/place/Abbasid-dynasty",
        "note": "Equinox Terr falls from 6.919078 to 6.000000. Britannica describes Abbasid temporal decentralization and local dynasties after the 9th century; retain as a plausible contraction.",
    },
    {
        "case_type": "contraction",
        "polity": "Susiana::IrSeleu",
        "time_t": -300,
        "source": "https://www.britannica.com/place/Seleucid-Empire",
        "note": "Equinox Terr falls from 6.556303 to 6.142233. Britannica documents Seleucid territorial losses in the 3rd century BCE; retain.",
    },
    {
        "case_type": "contraction",
        "polity": "Upper Egypt::EgPtol2",
        "time_t": -200,
        "source": "https://www.britannica.com/biography/Ptolemy-II-Philadelphus",
        "note": "Equinox Terr falls from 5.920645 to 5.571709. Historical context confirms a Ptolemaic state, but not enough evidence here to correct the Equinox territory measurement; retain provisionally.",
    },
    {
        "case_type": "contraction",
        "polity": "Upper Egypt::TnFatim",
        "time_t": 1000,
        "source": "https://www.britannica.com/place/Fatimid-dynasty",
        "note": "Equinox Terr falls from 6.146128 to 5.800029. Fatimid power faced provincial losses and growing instability in the 11th century; retain as plausible contraction.",
    },
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = pd.read_excel(root / "data/raw/Equinox_on_GitHub_June9_2022.xlsx", sheet_name="AggrSCWarAgriRelig")
    raw = raw.replace(-999, pd.NA)
    raw = raw[raw["uniq"].astype(str).str.lower().ne("n")].copy()
    raw["polity"] = raw["NGA"] + "::" + raw["PolID"]
    raw["Time"] = pd.to_numeric(raw["Time"])
    raw["Terr"] = pd.to_numeric(raw["Terr"])
    future = raw[["polity", "Time", "Terr"]].rename(columns={"Time": "next_time", "Terr": "next_Terr"})
    windows = raw.assign(next_time=raw["Time"] + 100).merge(
        future, on=["polity", "next_time"], how="left"
    )
    audited = pd.DataFrame(CASES).merge(
        windows[["polity", "Time", "Terr", "next_time", "next_Terr"]],
        left_on=["polity", "time_t"], right_on=["polity", "Time"], how="left",
    ).drop(columns="Time")
    audited["log10_change"] = audited["next_Terr"] - audited["Terr"]
    audited["ratio"] = 10 ** audited["log10_change"]
    audited["verified_correction"] = "none"
    audited.to_csv(root / "data" / "historical_case_audit.csv", index=False)
    print(audited[["case_type", "polity", "time_t", "Terr", "next_Terr", "ratio", "verified_correction"]].to_string(index=False))
    print(f"saved: {root / 'data/historical_case_audit.csv'}")
    print("No verified corrections made; the baseline labels remain unchanged.")


if __name__ == "__main__":
    main()