"""Project-wide constants: record lists, class names, annotation mapping.

Single source of truth. Nothing here is tunable; tunable values go in configs/.
"""

FS_HZ = 360

CLASS_NAMES = ["N", "S", "V", "F"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# AAMI EC57 style mapping from MIT-BIH beat symbols to the four target classes.
SYMBOL_TO_CLASS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}

# Beat symbols deliberately excluded (paced, fusion of paced and normal, unclassifiable).
EXCLUDED_BEAT_SYMBOLS = {"/", "f", "Q"}

# Everything else in the .atr files is a non-beat annotation (rhythm, noise, etc.).

DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

NSTDB_NOISE_RECORDS = ["bw", "em", "ma"]
