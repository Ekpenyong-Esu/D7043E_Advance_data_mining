# What is inside a MIT-BIH record

Every record in `mitdb/` is stored as three files with the same name and different endings. Record 101 is used as the example below. All values are taken from the real files.

Read in this order:

1. `101.hea`: the header. A small text file that describes the record.
2. `101.dat`: the signal. A binary file with the ECG values.
3. `101.atr`: the annotations. A binary file with one entry per heartbeat.

How they connect: the header tells wfdb how to read the other two. `wfdb.rdrecord` reads the header and the signal. `wfdb.rdann` reads the annotations. `src/ecg/data/records.py` calls both and packs the result into a `Record` object.

## 1. The header: 101.hea

This is plain text. You can open it in any editor.

```
101 2 360 650000
101.dat 212 200 11 1024 955 29832 0 MLII
101.dat 212 200 11 1024 992 19589 0 V1
# 75 F 1011 654 x1
# Diapres
```

Line 1 describes the record:

| value | meaning |
|---|---|
| 101 | record name |
| 2 | number of channels |
| 360 | samples per second |
| 650000 | samples per channel. 650000 / 360 = 1805 seconds, about 30 minutes |

Lines 2 and 3 describe one channel each:

| value | meaning |
|---|---|
| 101.dat | the file that holds the signal |
| 212 | storage format. Two 12-bit values packed into three bytes |
| 200 | gain. 200 stored units equal 1 millivolt |
| 11 | resolution of the recorder in bits |
| 1024 | zero level. A stored value of 1024 means 0 millivolts |
| 955 / 992 | the first stored value of that channel |
| 29832 / 19589 | checksum, used to detect a damaged file |
| 0 | block size, not used here |
| MLII / V1 | channel name |

The lines that start with `#` are comments. They say the patient was a 75-year-old woman and name her medicine.

The project reads the channel by name, MLII, not by position. Record 114 has MLII as the second channel, so reading by position would give the wrong lead there.

## 2. The signal: 101.dat

This is a binary file. In an editor it looks like nonsense. The first twelve bytes are:

```
bb33 e0bb 33e0 bb33 e0bb 33e0
```

Format 212 packs two 12-bit numbers into three bytes. wfdb unpacks them into whole numbers, called ADC counts. Here are the first three samples, one row per sample, one column per channel:

```
MLII  V1
 955  992
 955  992
 955  992
```

wfdb then converts to millivolts with the gain and zero level from the header:

```
millivolts = (stored value - zero level) / gain
           = (955 - 1024) / 200
           = -0.345
```

So the first samples in millivolts are:

```
MLII    V1
-0.345  -0.160
-0.345  -0.160
-0.345  -0.160
```

This array is `rec.p_signal` in the code. The "p" stands for physical units. Its shape is (650000, 2). The file is 1 950 000 bytes, which is 650000 samples times 2 channels times 1.5 bytes each.

## 3. The annotations: 101.atr

Also binary, and much smaller: 3768 bytes. It holds one small entry per annotation, not per sample. Record 101 has 1874 entries.

wfdb decodes each entry into a sample position and a symbol. The first five are:

| sample | seconds | symbol | meaning |
|---|---|---|---|
| 7 | 0.019 | + | rhythm change. The note "(N" means a normal rhythm starts here |
| 83 | 0.231 | N | normal beat |
| 396 | 1.100 | N | normal beat |
| 711 | 1.975 | N | normal beat |
| 1032 | 2.867 | N | normal beat |

The sample column is the position of the R-peak, the tall spike of the heartbeat. Divide by 360 to get seconds. The symbol column is the beat type written down by the cardiologists.

Not every entry is a heartbeat. The first one, `+`, is a rhythm mark. Other non-beat symbols you will see are `~` for noise and `|` for an artefact. The code in `src/ecg/data/labels.py` keeps only the beat symbols and maps them to the four classes N, S, V and F. The full mapping is in `results/label_mapping.json`.

In the code, `ann.sample` is the sample column and `ann.symbol` is the symbol column.

## Summary

| file | type | size for record 101 | what the project takes from it |
|---|---|---|---|
| 101.hea | text | 5 lines | sampling rate, channel names |
| 101.dat | binary | 1.95 MB | the ECG in millivolts, shape (650000, 2) |
| 101.atr | binary | 3.7 kB | 1874 beat positions and symbols |

To look at any record yourself:

```python
import wfdb
rec = wfdb.rdrecord("data/raw/mitdb/101")
ann = wfdb.rdann("data/raw/mitdb/101", "atr")
print(rec.fs, rec.sig_name, rec.p_signal.shape)
print(ann.sample[:5], ann.symbol[:5])
```
