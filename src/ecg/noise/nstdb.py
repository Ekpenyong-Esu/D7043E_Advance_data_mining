"""Load one pure-noise record (bw, em or ma) from the Noise Stress Test Database."""
from pathlib import Path

import numpy as np
import wfdb


def load_noise(raw_dir: str | Path, kind: str, channel: int = 0) -> np.ndarray:
    """Return one channel of the noise record ``kind`` as a 1-D array in millivolts."""
    return wfdb.rdrecord(str(Path(raw_dir) / kind)).p_signal[:, channel]
