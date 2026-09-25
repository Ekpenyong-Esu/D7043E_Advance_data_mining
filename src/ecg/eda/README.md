# ecg/eda/

Data Understanding (report section 2).

Start with `style.py`. It only uses `constants.py`. `stats.py` and `plots.py` build on it and on `ecg/data/`.

1. `style.py`: class colours and matplotlib style.
2. `stats.py`: functions that turn records into tables: symbol counts, class counts per record and per split, lead table, amplitude stats, noise-annotation counts.
3. `plots.py`: one function per report figure: raw signal, class examples, class distribution, record distribution, amplitude, noisy examples.
