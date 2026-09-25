"""Check that noise is added at the requested SNR. Skipped until ``snr.add_noise`` is written."""
import pytest

from ecg.noise.snr import add_noise, power


@pytest.mark.skip(reason="implement add_noise first")
def test_target_snr_is_reached(rng):
    pass
