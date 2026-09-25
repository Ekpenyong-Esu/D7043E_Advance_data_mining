"""Run the whole data preparation, from raw records to the .npz files.

For each record: read it, pick the lead, filter, map the labels, cut the beats,
normalise, and compute the RR features. Then stack the records of each split
and save them. Each step is a small function in its own module. This file only
calls them in the right order.
"""
from pathlib import Path

import numpy as np

from ecg.constants import CLASS_NAMES
from ecg.data import beats, channels, filters, labels, normalize, records, rr, splits, storage
from ecg.logging_utils import get_logger

log = get_logger(__name__)


def process_record(cfg: dict, record_id: int) -> dict:
    """Prepare one record. Returns a dict with X, y, rr, excluded counts and the edge-dropped count."""
    rec = records.read_record(cfg["paths"]["raw_mitdb"], record_id)
    x = channels.select_lead(rec, cfg["lead"])
    x = filters.apply_filters(x, rec.fs, cfg)

    # Map every annotation to a class. ``keep`` marks the real beats we use.
    y_all, keep, excluded = labels.map_symbols(rec.ann_symbols)
    peaks = rec.ann_samples[keep]

    # Cut a fixed window around each kept peak. Beats too close to the record edges are dropped,
    # so ``kept_idx`` tells us which peaks survived. We use it to keep labels and RR features aligned.
    windows, kept_idx = beats.extract_beats(
        x, peaks, cfg["beat"]["samples_before"], cfg["beat"]["samples_after"]
    )
    X = normalize.normalize_beats(windows, cfg)[:, None, :]  # add a channel axis: (n, 1, L)
    return {
        "X": X,
        "y": y_all[keep][kept_idx],
        "rr": rr.rr_features(peaks, rec.fs)[kept_idx],
        "excluded": excluded,
        "edge_dropped": int(len(peaks) - len(kept_idx)),
    }


def prepare_all_splits(cfg: dict) -> None:
    """Process every record, save one .npz per split, and write the summary tables to results/."""
    split_records = splits.build_splits(cfg["records"]["validation"])
    splits.assert_disjoint(split_records)
    out_dir = Path(cfg["paths"]["processed"]); out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["paths"]["results"]); res_dir.mkdir(parents=True, exist_ok=True)

    labels_per_split: dict[str, np.ndarray] = {}  # split name -> all its labels, for the stats table
    per_record: list[dict] = []                    # one row per record, for record_stats.csv
    for name, rec_ids in split_records.items():
        parts = [process_record(cfg, rid) for rid in rec_ids]

        # Collect the per-record numbers for the report.
        for rid, p in zip(rec_ids, parts):
            counts = {c: int((p["y"] == i).sum()) for i, c in enumerate(CLASS_NAMES)}
            per_record.append({"record": rid, "split": name, **counts, "total": len(p["y"]),
                               "edge_dropped": p["edge_dropped"],
                               **{f"excluded_{k}": v for k, v in p["excluded"].items()}})
            log.info("%s record %d: %s excluded=%s", name, rid, counts, p["excluded"])

        # Stack all records of this split into single arrays and save them.
        X = np.concatenate([p["X"] for p in parts])
        y = np.concatenate([p["y"] for p in parts])
        r = np.concatenate([p["rr"] for p in parts])
        ids = np.concatenate([np.full(len(p["y"]), rid) for rid, p in zip(rec_ids, parts)])
        storage.save_split(out_dir / f"{name}.npz", X, y, ids, r)
        labels_per_split[name] = y

    # Summary tables that the report quotes.
    stats = storage.split_statistics(labels_per_split, split_records)
    stats.to_csv(res_dir / "split_stats.csv", index=False)
    storage.record_statistics(per_record).to_csv(res_dir / "record_stats.csv", index=False)
    storage.write_label_mapping(res_dir / "label_mapping.json")
    log.info("\n%s", stats.to_string(index=False))
