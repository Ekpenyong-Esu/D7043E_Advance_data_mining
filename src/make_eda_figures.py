"""Entry point: make every EDA figure and table for report section 2.

Run with ``make eda``. Tables go to ``results/eda/``, figures to ``report/figures/``.
This file only wires things together. Statistics are in ``ecg.eda.stats``, plots in ``ecg.eda.plots``.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, not to a window
import numpy as np

from ecg.config import load_config
from ecg.constants import DS1, DS2, NSTDB_NOISE_RECORDS
from ecg.eda import plots, stats
from ecg.eda.style import apply_style
from ecg.noise.nstdb import load_noise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--out", default="report/figures")
    args = parser.parse_args()
    cfg = load_config(args.config); apply_style()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tables = Path(cfg["paths"]["results"]) / "eda"; tables.mkdir(parents=True, exist_ok=True)

    # 1. Load all 44 records and compute the tables.
    recs = stats.load_records(cfg["paths"]["raw_mitdb"], DS1 + DS2)
    cc = stats.class_counts(recs); totals = stats.class_totals(cc)
    amp = stats.amplitude_stats(recs, cfg["lead"])
    cc.to_csv(tables / "class_counts_per_record.csv"); totals.to_csv(tables / "class_totals.csv")
    stats.lead_table(recs).to_csv(tables / "leads.csv"); amp.to_csv(tables / "amplitude.csv")
    stats.symbol_counts(recs).to_csv(tables / "symbol_counts.csv")
    stats.noise_annotation_counts(recs).to_csv(tables / "noise_marks.csv")

    # 2. Draw the figures. Records 203 and 207 are used as noisy examples.
    b, a = cfg["beat"]["samples_before"], cfg["beat"]["samples_after"]
    noise = {k: load_noise(cfg["paths"]["raw_nstdb"], k) for k in NSTDB_NOISE_RECORDS}
    figures = {
        "eda_raw_ecg": plots.raw_signal(recs[100]),
        "prep_beat_window": plots.beat_window(recs[100], cfg["lead"], b, a),
        "eda_class_examples": plots.class_examples(recs, cfg["lead"], b, a),
        # DS1 records only, so the test set stays unseen.
        "eda_class_context": plots.class_in_context(recs, cfg["lead"], {"N": 101, "S": 209, "V": 119, "F": 208}, b, a, seconds=4.0),
        "eda_class_distribution": plots.class_distribution(totals),
        "eda_record_distribution": plots.record_distribution(cc),
        "eda_amplitude": plots.amplitude_distribution(amp),
        "eda_noisy": plots.noisy_examples([recs[203], recs[207]], noise, cfg["lead"], cfg["fs"]),
    }

    # 3. The augmentation figure needs a processed beat, so it is only drawn after ``make prepare``.
    processed = Path(cfg["paths"]["processed"]) / "train.npz"
    if processed.exists():
        d = np.load(processed); beat = d["X"][np.flatnonzero(d["y"] == 0)[100], 0]  # the 101st N beat
        figures["aug_amplitude_scale"] = plots.amplitude_scale_example(beat, cfg["fs"], b)

    for name, fig in figures.items():
        fig.savefig(out / f"{name}.png")
    print(f"wrote {len(figures)} figures to {out} and tables to {tables}")


if __name__ == "__main__":
    main()
