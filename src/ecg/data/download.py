"""Download a PhysioNet database with wfdb. Safe to run again: it skips a folder that already has files."""
from pathlib import Path

import wfdb


def download_database(name: str, target: str | Path) -> None:
    """Download the database ``name`` (for example ``mitdb``) into ``target``, unless it is already there."""
    target = Path(target)
    if target.exists() and any(target.iterdir()):
        return  # already downloaded
    wfdb.dl_database(name, str(target))
