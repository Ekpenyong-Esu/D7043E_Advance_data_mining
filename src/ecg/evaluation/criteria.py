"""Check the success criteria of report section 1.4 against the results (report section 5.6)."""
import pandas as pd

TARGETS = {"macro_f1": 0.70, "recall_S": 0.50, "recall_F": 0.50, "recall_V": 0.85, "noise_drop": 0.15,
           "params": 100_000, "int8_kb": 200, "int8_loss": 0.03}


def _status(ok: list[bool]) -> str:
    return "Achieved" if all(ok) else "Partially achieved" if any(ok) else "Not achieved"


def success_criteria(metrics: pd.DataFrame, noise: pd.DataFrame, evidence: pd.DataFrame, weight_bytes: int,
                     int8: str = "cnn_int8", fp32: str = "cnn_fp32") -> pd.DataFrame:
    """One row per criterion: target, measured result and Achieved / Partially achieved / Not achieved.

    ``metrics`` is results/metrics.csv (DS2), ``noise`` is results/noise_results.csv,
    ``evidence`` the table of section 6.3 and ``weight_bytes`` the INT8 weight size from the synthesiser.
    """
    m = metrics.set_index("model")
    d, f = m.loc[int8], m.loc[fp32]
    n = noise[noise["model"] == int8]
    clean = float(n.loc[n["noise"] == "none", "macro_f1"].iloc[0])
    at6 = float(n.loc[n["snr_db"].astype(str) == "6", "macro_f1"].mean())
    drop = clean - at6
    passes = evidence["Result"].head(4).eq("PASS").tolist()
    loss = float(f["macro_f1"] - d["macro_f1"])
    rows = [
        ("Predictive performance", f"macro-F1 $\\geq$ {TARGETS['macro_f1']:.2f} on DS2", f"{d['macro_f1']:.3f}",
         _status([d["macro_f1"] >= TARGETS["macro_f1"]])),
        ("Minority class performance", "recall S and F $\\geq$ 0.50, V $\\geq$ 0.85",
         f"S {d['recall_S']:.2f}, F {d['recall_F']:.2f}, V {d['recall_V']:.2f}",
         _status([d["recall_S"] >= TARGETS["recall_S"], d["recall_F"] >= TARGETS["recall_F"], d["recall_V"] >= TARGETS["recall_V"]])),
        ("Noise robustness", f"macro-F1 drop clean to 6 dB $\\leq$ {TARGETS['noise_drop']:.2f}",
         f"{clean:.3f} to {at6:.3f} (drop {drop:.3f})", _status([drop <= TARGETS["noise_drop"]])),
        ("Model complexity", "$\\leq$ 100\\,000 parameters, $\\leq$ 200 kB INT8",
         f"{int(d['n_params']):,}".replace(",", "\\,") + f" parameters, {weight_bytes / 1024:.1f} kB",
         _status([d["n_params"] <= TARGETS["params"], weight_bytes / 1024 <= TARGETS["int8_kb"]])),
        ("MAX78002 compatibility", "PASS on all four toolchain checks", f"{sum(passes)} of 4 PASS", _status(passes)),
        ("Quantisation loss", f"INT8 macro-F1 within {TARGETS['int8_loss']:.2f} of FP32", f"loss {loss:.3f}",
         _status([loss <= TARGETS["int8_loss"]])),
    ]
    return pd.DataFrame(rows, columns=["Criterion", "Target", "Result on DS2", "Status"])
