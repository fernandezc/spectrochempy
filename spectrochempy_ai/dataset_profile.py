"""
Dataset profiling for template recommendation.

Extracts structural characteristics from a spectral dataset file without
assuming knowledge of its scientific context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SelectedDataset:
    """Selection result for a dataset source that may contain several objects."""

    dataset: Any | None
    source_was_multi_object: bool = False
    source_object_count: int | None = None
    selected_object_index: int | None = None
    selected_object_name: str | None = None
    selection_note: str | None = None
    error: str | None = None


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
    source_was_multi_object: bool = False
    source_object_count: int | None = None
    selected_object_index: int | None = None
    selected_object_name: str | None = None
    selection_note: str | None = None
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


def _select_dataset(result: Any) -> SelectedDataset:
    """Resolve a reader result to one dataset, selecting deterministically if needed."""
    if result is None:
        return SelectedDataset(dataset=None, error="File format not recognised")

    if not isinstance(result, list):
        return SelectedDataset(dataset=result)

    if len(result) == 0:
        return SelectedDataset(
            dataset=None,
            source_was_multi_object=True,
            source_object_count=0,
            error="Multi-object file is empty",
        )

    if len(result) == 1:
        return SelectedDataset(dataset=result[0])

    candidates = (
        result.filter_by_ndim(2)
        if hasattr(result, "filter_by_ndim")
        else [obj for obj in result if hasattr(obj, "ndim") and obj.ndim == 2]
    )
    if not candidates:
        return SelectedDataset(
            dataset=None,
            source_was_multi_object=True,
            source_object_count=len(result),
            error="Multi-object file contains no suitable 2D dataset",
        )

    selected = (
        candidates.select_largest()
        if hasattr(candidates, "select_largest")
        else max(candidates, key=lambda ds: ds.size if hasattr(ds, "size") else 0)
    )
    index = next((i for i, obj in enumerate(result) if obj is selected), None)
    name = getattr(selected, "name", None) or None

    details = []
    if index is not None:
        details.append(f"index {index}")
    if name:
        details.append(f"name '{name}'")
    selection_note = (
        f"Multi-object file ({len(result)} objects): automatically selected the "
        f"largest 2D dataset"
    )
    if details:
        selection_note += f" ({', '.join(details)})"

    return SelectedDataset(
        dataset=selected,
        source_was_multi_object=True,
        source_object_count=len(result),
        selected_object_index=index,
        selected_object_name=name,
        selection_note=selection_note,
    )


def resolve_dataset_source(path: str | Path) -> SelectedDataset:
    """Read *path* and resolve it to a single dataset plus selection metadata."""
    src = Path(path)
    try:
        import spectrochempy as scp  # noqa: PLC0415

        return _select_dataset(scp.read(src))
    except Exception as exc:
        return SelectedDataset(dataset=None, error=str(exc))


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

    selection = resolve_dataset_source(src)
    if selection.error or selection.dataset is None:
        return DatasetProfile(
            path=src.resolve(),
            readable=False,
            summary=f"Cannot read {src.suffix} file",
            source_was_multi_object=selection.source_was_multi_object,
            source_object_count=selection.source_object_count,
            selected_object_index=selection.selected_object_index,
            selected_object_name=selection.selected_object_name,
            selection_note=selection.selection_note,
            error=selection.error,
        )
    dataset = selection.dataset

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
    if selection.source_was_multi_object:
        parts.append(f"multi-object ({selection.source_object_count} objects)")
        selection_bits: list[str] = []
        if selection.selected_object_index is not None:
            selection_bits.append(f"selected index {selection.selected_object_index}")
        if selection.selected_object_name:
            selection_bits.append(f"name '{selection.selected_object_name}'")
        if selection_bits:
            parts.append(", ".join(selection_bits))
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
        source_was_multi_object=selection.source_was_multi_object,
        source_object_count=selection.source_object_count,
        selected_object_index=selection.selected_object_index,
        selected_object_name=selection.selected_object_name,
        selection_note=selection.selection_note,
        summary=summary,
    )
