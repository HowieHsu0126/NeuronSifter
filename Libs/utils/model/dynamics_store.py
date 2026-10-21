from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from Libs.utils.data.raw_data_registry import ensure_directory, write_json


def chunk_store_manifest_path(store_path: Path) -> Path:
    return store_path / "store_manifest.json"


def chunk_filename(chunk_index: int) -> str:
    return f"chunk_{chunk_index:06d}.npy"


def zarr_array_metadata_path(store_path: Path, array_name: str) -> Path:
    return store_path / array_name / ".zarray"


def zarr_chunk_filename(chunk_index: tuple[int, ...]) -> str:
    return ".".join(str(index) for index in chunk_index)


def zarr_typestr(dtype: np.dtype) -> str:
    return np.dtype(dtype).newbyteorder("<").str


def chunk_slices(shape: tuple[int, ...], chunks: tuple[int, ...]):
    chunk_ranges = [range(int(math.ceil(axis_size / chunk_size))) for axis_size, chunk_size in zip(shape, chunks)]
    for chunk_index in product(*chunk_ranges):
        slices = tuple(
            slice(index * chunk_size, min((index + 1) * chunk_size, axis_size))
            for index, axis_size, chunk_size in zip(chunk_index, shape, chunks)
        )
        yield chunk_index, slices


def initialize_zarr_store(store_path: Path, metadata: dict[str, object]) -> None:
    ensure_directory(store_path)
    write_json(store_path / ".zgroup", {"zarr_format": 2})
    write_json(store_path / ".zattrs", metadata)


def write_zarr_array(
    store_path: Path,
    array_name: str,
    array: np.ndarray,
    chunks: tuple[int, ...],
    attrs: dict[str, object] | None = None,
) -> None:
    if array.ndim != len(chunks):
        raise ValueError(f"Zarr chunks for {array_name} must match ndim {array.ndim}, got {len(chunks)}.")
    array_root = store_path / array_name
    ensure_directory(array_root)
    dtype = np.dtype(array.dtype)
    write_json(
        array_root / ".zarray",
        {
            "zarr_format": 2,
            "shape": list(array.shape),
            "chunks": [int(value) for value in chunks],
            "dtype": zarr_typestr(dtype),
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
        },
    )
    write_json(array_root / ".zattrs", {} if attrs is None else attrs)
    for chunk_index, slices in chunk_slices(tuple(array.shape), tuple(int(value) for value in chunks)):
        chunk = np.ascontiguousarray(array[slices].astype(dtype, copy=False))
        chunk.tofile(array_root / zarr_chunk_filename(chunk_index))

def write_zarr_array_from_loader(
    store_path: Path,
    array_name: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    chunk_loader,
    attrs: dict[str, object] | None = None,
) -> None:
    if len(shape) != len(chunks):
        raise ValueError(f"Zarr chunks for {array_name} must match ndim {len(shape)}, got {len(chunks)}.")
    array_root = store_path / array_name
    ensure_directory(array_root)
    dtype = np.dtype(dtype)
    write_json(
        array_root / ".zarray",
        {
            "zarr_format": 2,
            "shape": [int(value) for value in shape],
            "chunks": [int(value) for value in chunks],
            "dtype": zarr_typestr(dtype),
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
        },
    )
    write_json(array_root / ".zattrs", {} if attrs is None else attrs)
    for chunk_index, slices in chunk_slices(tuple(shape), tuple(int(value) for value in chunks)):
        chunk = np.asarray(chunk_loader(slices), dtype=dtype)
        expected_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in slices)
        if chunk.shape != expected_shape:
            raise ValueError(f"Chunk loader for {array_name} returned shape {chunk.shape}, expected {expected_shape}.")
        np.ascontiguousarray(chunk).tofile(array_root / zarr_chunk_filename(chunk_index))


def read_zarr_array_metadata(store_path: Path, array_name: str) -> dict[str, object]:
    return json.loads(zarr_array_metadata_path(store_path, array_name).read_text(encoding="utf-8"))


def load_zarr_full_array(store_path: Path, array_name: str) -> np.ndarray:
    metadata = read_zarr_array_metadata(store_path, array_name)
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(str(metadata["dtype"]))
    array = np.zeros(shape, dtype=dtype)
    array_root = store_path / array_name
    for chunk_index, slices in chunk_slices(shape, chunks):
        chunk_path = array_root / zarr_chunk_filename(chunk_index)
        chunk_shape = tuple(slice_item.stop - slice_item.start for slice_item in slices)
        array[slices] = np.fromfile(chunk_path, dtype=dtype).reshape(chunk_shape)
    return array


def load_zarr_rows(store_path: Path, array_name: str, row_indices: np.ndarray) -> np.ndarray:
    metadata = read_zarr_array_metadata(store_path, array_name)
    shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(str(metadata["dtype"]))
    row_chunk_size = chunks[0]
    row_indices_array = np.asarray(row_indices, dtype=np.int64)
    output = np.zeros((len(row_indices_array), *shape[1:]), dtype=dtype)
    array_root = store_path / array_name
    positions_by_row_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_index, row_index in enumerate(row_indices_array.tolist()):
        chunk_index_prefix = int(row_index // row_chunk_size)
        row_within_chunk = int(row_index % row_chunk_size)
        positions_by_row_chunk.setdefault(chunk_index_prefix, []).append((output_index, row_within_chunk))
    for chunk_index_prefix, positions in positions_by_row_chunk.items():
        for tail_chunk_index in product(*[range(int(math.ceil(axis_size / chunk_size))) for axis_size, chunk_size in zip(shape[1:], chunks[1:])]):
            chunk_index = (chunk_index_prefix, *tail_chunk_index)
            chunk_slices_tuple = tuple(
                slice(index * chunk_size, min((index + 1) * chunk_size, axis_size))
                for index, axis_size, chunk_size in zip(chunk_index, shape, chunks)
            )
            chunk_shape = tuple(slice_item.stop - slice_item.start for slice_item in chunk_slices_tuple)
            chunk = np.fromfile(array_root / zarr_chunk_filename(chunk_index), dtype=dtype).reshape(chunk_shape)
            output_slices = tuple(slice_item for slice_item in chunk_slices_tuple[1:])
            for output_index, row_within_chunk in positions:
                output[(output_index, *output_slices)] = chunk[row_within_chunk]
    return output


def read_store_manifest(store_path: Path) -> dict[str, object]:
    manifest_path = chunk_store_manifest_path(store_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def initialize_chunk_store(
    store_path: Path,
    axis_names: list[str],
    coordinate_payload: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    ensure_directory(store_path)
    ensure_directory(store_path / "coordinates")
    write_json(
        chunk_store_manifest_path(store_path),
        {
            "format_name": "neuronsifter_chunk_store",
            "axis_names": axis_names,
            "coordinates": {name: f"coordinates/{name}.npy" for name in coordinate_payload},
            "arrays": {},
            "metadata": metadata,
        },
    )
    for coordinate_name, coordinate_values in coordinate_payload.items():
        np.save(store_path / "coordinates" / f"{coordinate_name}.npy", np.asarray(coordinate_values), allow_pickle=False)


def register_array(
    store_path: Path,
    array_name: str,
    shape: tuple[int, ...],
    dtype_name: str,
    chunk_size: int,
) -> None:
    manifest = read_store_manifest(store_path)
    arrays = dict(manifest["arrays"])
    arrays[array_name] = {
        "shape": list(shape),
        "dtype": dtype_name,
        "chunk_size": int(chunk_size),
        "chunk_count": int(math.ceil(shape[0] / max(chunk_size, 1))),
        "relative_root": f"arrays/{array_name}",
    }
    manifest["arrays"] = arrays
    write_json(chunk_store_manifest_path(store_path), manifest)


def write_chunked_array(
    store_path: Path,
    array_name: str,
    array: np.ndarray,
    chunk_size: int,
) -> None:
    if array.ndim < 1:
        raise ValueError("Chunked arrays must have at least one dimension.")
    chunk_root = store_path / "arrays" / array_name
    ensure_directory(chunk_root)
    register_array(store_path, array_name, tuple(array.shape), str(array.dtype), int(chunk_size))
    for start_index in range(0, array.shape[0], chunk_size):
        end_index = min(start_index + chunk_size, array.shape[0])
        chunk_index = start_index // chunk_size
        np.save(chunk_root / chunk_filename(chunk_index), array[start_index:end_index], allow_pickle=False)


def write_chunk_sequence(
    store_path: Path,
    array_name: str,
    shape: tuple[int, ...],
    dtype_name: str,
    chunk_size: int,
    chunk_iterable,
) -> None:
    if len(shape) < 1:
        raise ValueError("Chunked arrays must have at least one dimension.")
    chunk_root = store_path / "arrays" / array_name
    ensure_directory(chunk_root)
    register_array(store_path, array_name, shape, dtype_name, int(chunk_size))
    written_rows = 0
    expected_tail_shape = tuple(int(value) for value in shape[1:])
    for chunk_index, chunk in enumerate(chunk_iterable):
        chunk_array = np.asarray(chunk)
        if chunk_array.ndim != len(shape):
            raise ValueError(f"Chunk {chunk_index} for {array_name} has ndim {chunk_array.ndim}, expected {len(shape)}.")
        if tuple(int(value) for value in chunk_array.shape[1:]) != expected_tail_shape:
            raise ValueError(f"Chunk {chunk_index} for {array_name} has tail shape {chunk_array.shape[1:]}, expected {expected_tail_shape}.")
        if chunk_array.shape[0] > int(chunk_size):
            raise ValueError(f"Chunk {chunk_index} for {array_name} has {chunk_array.shape[0]} rows, exceeding chunk size {chunk_size}.")
        written_rows += int(chunk_array.shape[0])
        np.save(chunk_root / chunk_filename(chunk_index), chunk_array.astype(dtype_name, copy=False), allow_pickle=False)
    if written_rows != int(shape[0]):
        raise ValueError(f"Chunk sequence for {array_name} wrote {written_rows} rows, expected {shape[0]}.")


def load_coordinate(store_path: Path, coordinate_name: str) -> np.ndarray:
    manifest = read_store_manifest(store_path)
    coordinate_path = store_path / str(manifest["coordinates"][coordinate_name])
    return np.load(coordinate_path, allow_pickle=False)


def load_chunked_rows(store_path: Path, array_name: str, row_indices: np.ndarray) -> np.ndarray:
    manifest = read_store_manifest(store_path)
    array_manifest = manifest["arrays"][array_name]
    chunk_root = store_path / str(array_manifest["relative_root"])
    chunk_size = int(array_manifest["chunk_size"])
    requested_indices = np.asarray(row_indices, dtype=np.int64).tolist()
    if not requested_indices:
        return np.empty((0,), dtype=np.dtype(str(array_manifest["dtype"])))
    output_rows: list[np.ndarray | None] = [None] * len(requested_indices)
    positions_by_chunk: dict[int, list[tuple[int, int]]] = {}
    for output_position, row_index in enumerate(requested_indices):
        chunk_index = int(row_index // chunk_size)
        within_chunk_index = int(row_index % chunk_size)
        positions_by_chunk.setdefault(chunk_index, []).append((output_position, within_chunk_index))
    for chunk_index, positions in positions_by_chunk.items():
        chunk = np.load(chunk_root / chunk_filename(chunk_index), allow_pickle=False)
        for output_position, within_chunk_index in positions:
            output_rows[output_position] = chunk[within_chunk_index]
    if any(row is None for row in output_rows):
        raise ValueError(f"Chunked row loading for {array_name} did not populate every requested output row.")
    return np.stack([np.asarray(row) for row in output_rows], axis=0)


def load_full_array(store_path: Path, array_name: str) -> np.ndarray:
    manifest = read_store_manifest(store_path)
    array_manifest = manifest["arrays"][array_name]
    chunk_root = store_path / str(array_manifest["relative_root"])
    chunk_count = int(array_manifest["chunk_count"])
    chunks = [np.load(chunk_root / chunk_filename(chunk_index), allow_pickle=False) for chunk_index in range(chunk_count)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
