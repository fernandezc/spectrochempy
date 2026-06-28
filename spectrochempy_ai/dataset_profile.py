"""
Dataset profiling for template recommendation.

Extracts structural characteristics from a spectral dataset file without
assuming knowledge of its scientific context.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass
class DatasetProfile:
    """
    Structural characteristics of a spectral dataset file.

    Fields are ``None`` when the file could not be read or the property
    is not applicable.
    """

    path: Path
    readable: bool
    ndim: int | None = None
    shape: tuple[int, ...] | None = None
    dims: tuple[str, ...] | None = None
    has_continuous_x: bool | None = None
    x_unit: str | None = None
    n_observations: int | None = None
    n_variables: int | None = None
    summary: str = ""
    error: str | None = None

    @property
    def is_1d(self) -> bool:
        return self.ndim == 1

    @property
    def is_2d(self) -> bool:
        return self.ndim == 2

    @property
    def is_spectral(self) -> bool:
        """Heuristic: dataset has a continuous x coordinate."""
        return self.has_continuous_x is True


def profile_dataset(path: str | Path) -> DatasetProfile:
    """
    Read a spectral dataset file and return its structural profile.

    This function is robust: it catches read errors and returns a minimal
    profile with ``readable=False`` and an ``error`` message rather than
    crashing.
    """
    src = Path(path)

    if not src.exists():
        return DatasetProfile(
            path=src.resolve(),
            readable=False,
            summary=f"File not found: {src}",
            error="File does not exist",
        )

    try:
        import spectrochempy as scp

        dataset = scp.read(src)
        if dataset is None:
            return DatasetProfile(
                path=src.resolve(),
                readable=False,
                summary=f"Cannot read {src.suffix} file",
                error="File format not recognised",
            )
    except Exception as exc:
        return DatasetProfile(
            path=src.resolve(),
            readable=False,
            summary=f"Cannot read {src.suffix} file",
            error=str(exc),
        )

    # scp.read may return a list; unwrap single-element case
    if isinstance(dataset, list):
        if len(dataset) == 1:
            dataset = dataset[0]
        else:
            return DatasetProfile(
                path=src.resolve(),
                readable=False,
                summary=f"Read returned {len(dataset)} datasets",
                error="Multiple datasets returned; single expected",
            )

    ndim = dataset.ndim
    shape = tuple(dataset.shape) if dataset.shape else ()
    dims = tuple(dataset.dims) if hasattr(dataset, "dims") and dataset.dims else ()

    # Coordinate introspection
    has_continuous_x: bool | None = None
    x_unit: str | None = None
    try:
        x_coord = dataset.coord("x") if hasattr(dataset, "coord") else None
        if x_coord is not None and hasattr(x_coord, "has_data"):
            has_continuous_x = bool(x_coord.has_data)
            if hasattr(x_coord, "linear") and x_coord.linear:
                has_continuous_x = True
            if hasattr(x_coord, "units"):
                x_unit = str(x_coord.units) if x_coord.units else None
    except Exception:
        has_continuous_x = None

    # Observations vs variables
    n_obs: int | None = None
    n_vars: int | None = None
    if ndim == 2 and len(shape) == 2:
        n_obs = shape[0]
        n_vars = shape[1]
    elif ndim == 1 and len(shape) == 1:
        n_obs = 1
        n_vars = shape[0]

    # Human-readable summary
    parts: list[str] = []
    if ndim is not None:
        parts.append(f"{ndim}D")
    if shape:
        parts.append(f"shape {shape}")
    if dims:
        parts.append(f"dims {dims}")
    if x_unit:
        parts.append(f"x: {x_unit}")
    summary = ", ".join(parts) if parts else f"{src.name}"

    return DatasetProfile(
        path=src.resolve(),
        readable=True,
        ndim=ndim,
        shape=shape,
        dims=dims,
        has_continuous_x=has_continuous_x,
        x_unit=x_unit,
        n_observations=n_obs,
        n_variables=n_vars,
        summary=summary,
    )
