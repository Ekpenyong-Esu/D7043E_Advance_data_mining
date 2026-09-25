"""Fixed facts about the project: record lists, class names, symbol mapping.

Nothing here is a setting you tune. Settings live in configs/.
Every other module imports these values instead of writing its own copy.
"""

FS_HZ = 360  # sampling rate in Hz: the ECG has 360 samples per second

# The four target classes, in the order the model outputs them.
# N = normal, S = supraventricular, V = ventricular, F = fusion.
CLASS_NAMES = ["N", "S", "V", "F"]

# Class name -> class index, for example {"N": 0, "S": 1, "V": 2, "F": 3}.
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# MIT-BIH uses many beat symbols. The AAMI EC57 standard groups them into
# the four classes above. This is that grouping.
SYMBOL_TO_CLASS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",   # normal and bundle branch block beats
    "A": "S", "a": "S", "J": "S", "S": "S",             # beats that start above the ventricles
    "V": "V", "E": "V",                                 # beats that start in the ventricles
    "F": "F",                                           # fusion of a ventricular and a normal beat
}

# Beat symbols we drop on purpose:
#   "/" paced beat, "f" fusion of paced and normal, "Q" unclassifiable.
EXCLUDED_BEAT_SYMBOLS = {"/", "f", "Q"}

# Any other symbol in the .atr files is not a beat (rhythm change, noise, ...).

# The inter-patient split from de Chazal et al. (2004).
# DS1 is for training and validation. DS2 is the final test set.
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# Pure-noise records in the Noise Stress Test Database:
# bw = baseline wander, em = electrode motion, ma = muscle artefact.
NSTDB_NOISE_RECORDS = ["bw", "em", "ma"]
