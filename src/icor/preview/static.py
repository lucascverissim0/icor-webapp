"""Safe compiled-asset resolution for the same-origin preview service."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def resolve_asset(asset_root: Path, request_path: str) -> Path | None:
    """Return an existing in-root regular file, rejecting ambiguous paths."""
    if (
        not request_path
        or "\x00" in request_path
        or "\\" in request_path
        or "%" in request_path
    ):
        return None
    relative = PurePosixPath(request_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    try:
        root = asset_root.resolve(strict=True)
        candidate = root.joinpath(*relative.parts).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate
