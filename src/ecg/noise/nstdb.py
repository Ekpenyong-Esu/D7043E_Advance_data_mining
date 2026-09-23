"""Load the pure-noise records bw / em / ma from the NSTDB."""
from pathlib import Path

import numpy as np
import wfdb


def load_noise(raw_dir: str | Path, kind: str, channel: int = 0) -> np.ndarray:
    return wfdb.rdrecord(str(Path(raw_dir) / kind)).p_signal[:, channel]
