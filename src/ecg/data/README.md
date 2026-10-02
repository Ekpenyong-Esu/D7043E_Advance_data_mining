# ecg/data/

Data Preparation (report section 3).

Start with `records.py`. The first five files import nothing from the project, so read them first, in the order the data flows. `pipeline.py`, which calls all of them, comes last.

1. `records.py`: read one record: signal, lead names, annotation positions and symbols.
2. `filters.py`: baseline removal (median or high-pass) and optional low-pass.
3. `beats.py`: fixed 256-sample windows around annotated R-peaks; drops beats at the record edges.
4. `normalize.py`: per-beat z-score or max-abs, then clipping.
5. `rr.py`: pre and post RR intervals and local ratio.
6. `channels.py`: select the lead by name (record 114 has MLII in the second column). Uses `records.py`.
7. `labels.py`: MIT-BIH symbol to N/S/V/F index; counts every excluded symbol. Uses `constants.py`.
8. `splits.py`: record-level train/val/test assignment with a disjointness check. Uses `constants.py`.
9. `storage.py`: `.npz` write, split statistics table, label-mapping JSON. Uses `constants.py`.
10. `pipeline.py`: runs all the stages above for every record and every split.
