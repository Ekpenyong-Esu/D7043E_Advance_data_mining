# data/raw/

PhysioNet databases as downloaded by `make data` (`src/ecg/data/download.py`).

Start with `mitdb/`.

1. `mitdb/`: MIT-BIH Arrhythmia Database. 48 records, each with `NNN.dat` (signal), `NNN.hea` (header) and `NNN.atr` (beat annotations).
2. `nstdb/`: Noise Stress Test Database. Only the pure noise records `bw`, `em` and `ma` are used. The pre-mixed `118e*` and `119e*` records are not used, because records 118 and 119 are in DS1.

`RECORD_FILES.md` in this folder shows what the three files of one record contain, with record 101 as the example.

How they connect: `src/ecg/data/records.py` reads one `mitdb` record and its annotations. `src/ecg/noise/nstdb.py` reads the `nstdb` noise records. Both are used by the EDA and by the pipeline.

Behind a proxy, set `REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt` before downloading.
