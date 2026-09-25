"""Read the quantize.py and ai8xize.py logs and fill the PASS/FAIL table for report section 6.3. Not written yet."""


def build_evidence_table(quant_log: str, synth_log: str, synth_dir: str) -> dict[str, str]:
    """Return a dict: requirement -> "PASS", "FAIL" or a number (for the memory rows)."""
    raise NotImplementedError
