"""Find and drive the two Analog Devices repositories.

``ai8x-training`` provides ``ai8x.py``, the layer wrappers the CNN is built with.
``ai8x-synthesis`` provides ``quantize.py`` (FP32 checkpoint to INT8) and ``ai8xize.py``
(INT8 checkpoint plus network YAML to C code for the MAX78002). Both are cloned by
``make toolchain`` into the folders named in the ``toolchain:`` block of the config.
The synthesis scripts run in their own virtual environment with the versions they pin.
"""
import subprocess
import sys
from pathlib import Path

MAX78002 = 87  # the ai8x device number of the MAX78002


class ToolchainMissing(FileNotFoundError):
    """Raised when a repository or its virtual environment is not installed."""


def _repo(cfg: dict, key: str, must_have: str) -> Path:
    path = Path(cfg["toolchain"][key])
    if not (path / must_have).exists():
        raise ToolchainMissing(f"{path / must_have} not found. Run `make toolchain` first.")
    return path


def import_ai8x(cfg: dict, simulate: bool = False):
    """Import ``ai8x`` from ai8x-training and select the MAX78002.

    ``simulate=False`` is FP32 (and QAT) behaviour. ``simulate=True`` runs a quantised
    checkpoint with integer arithmetic, as the accelerator does.
    """
    path = _repo(cfg, "training", "ai8x.py")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import ai8x

    ai8x.set_device(MAX78002, simulate, False, verbose=False)
    return ai8x


def _run(cmd: list[str], cwd: Path, log_path: str | Path) -> bool:
    """Run a command, write its command line and output to ``log_path``, return True on success."""
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(f"$ {' '.join(cmd)}\n\n{r.stdout}\n{r.stderr}\n\nexit code: {r.returncode}\n")
    return r.returncode == 0


def run_quantize(cfg: dict, src: str | Path, dst: str | Path, log_path: str | Path) -> bool:
    """``quantize.py``: turn a checkpoint into the INT8 checkpoint ``dst``. Returns True on success.

    Without ``--clip-method`` the script uses the QAT information in the checkpoint, or, for
    an FP32 checkpoint, picks a per-layer scale from the weight range (post-training quantisation).
    """
    repo = _repo(cfg, "synthesis", "quantize.py")
    py = _repo(cfg, "synthesis", "venv/bin/python") / "venv/bin/python"
    cmd = [str(py), "quantize.py", str(Path(src).resolve()), str(Path(dst).resolve()), "--device", "MAX78002", "-v"]
    return _run(cmd, repo, log_path) and Path(dst).exists()


def run_synthesis(cfg: dict, checkpoint: str | Path, network_yaml: str | Path, sample_input: str | Path,
                  out_dir: str | Path, prefix: str, log_path: str | Path) -> bool:
    """``ai8xize.py``: generate the C project for the MAX78002 into ``out_dir/prefix``. Returns True on success."""
    repo = _repo(cfg, "synthesis", "ai8xize.py")
    py = _repo(cfg, "synthesis", "venv/bin/python") / "venv/bin/python"
    cmd = [str(py), "ai8xize.py", "--verbose", "--log", "--overwrite",
           "--test-dir", str(Path(out_dir).resolve()), "--prefix", prefix,
           "--checkpoint-file", str(Path(checkpoint).resolve()),
           "--config-file", str(Path(network_yaml).resolve()),
           "--sample-input", str(Path(sample_input).resolve()),
           "--device", "MAX78002", "--compact-data", "--mexpress", "--softmax"]
    return _run(cmd, repo, log_path)
