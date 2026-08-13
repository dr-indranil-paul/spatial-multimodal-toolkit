"""SHA-256 calculation and file integrity verification utilities."""

import hashlib
from pathlib import Path
from typing import Union


def calculate_sha256(filepath: Union[str, Path], chunk_size: int = 65536) -> str:
    """Calculate the SHA-256 hex digest of a file on disk."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest().lower()


def verify_file_checksum(filepath: Union[str, Path], expected_sha256: str) -> bool:
    """Verify that a file's SHA-256 matches the expected digest."""
    actual = calculate_sha256(filepath)
    return actual == expected_sha256.lower()
