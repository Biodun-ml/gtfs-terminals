from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from gtfs_terminals.geo import chord_to_km, km_to_chord, unit_xyz

MATCH_FIELDS = [
    pa.field("terminal_feed_id", pa.string()),
    pa.field("terminal_stop_id", pa.string()),
    pa.field("terminal_name", pa.string()),
    pa.field("distance_km", pa.float64()),
]


class TerminalIndex:
    # A KD-tree over raw (lat, lon) treats degrees as planar: a degree of longitude is
    # 111 km at the equator and 0 km at the poles, and the antimeridian splits neighbours.
    # On 3D unit vectors Euclidean (chord) distance is monotonic in great-circle distance,
    # so the nearest chord is the nearest terminal everywhere on the globe.
    def __init__(self, terminals: pd.DataFrame):
        if terminals.empty:
            raise ValueError("no terminals to index")
        self.terminals = terminals.reset_index(drop=True)
        self._tree = cKDTree(
            unit_xyz(self.terminals["lat"].to_numpy(), self.terminals["lon"].to_numpy())
        )

    def query(
        self, lat: np.ndarray, lon: np.ndarray, max_km: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        distance_km = np.full(len(lat), np.nan)
        index = np.full(len(lat), -1, dtype=np.int64)

        valid = np.isfinite(lat) & np.isfinite(lon)
        bound = np.inf if max_km is None else float(km_to_chord(max_km))
        chord, found = self._tree.query(
            unit_xyz(lat[valid], lon[valid]), k=1, distance_upper_bound=bound, workers=-1
        )
        hit = np.isfinite(chord)
        rows = np.flatnonzero(valid)[hit]
        distance_km[rows] = chord_to_km(chord[hit])
        index[rows] = found[hit]
        return distance_km, index


def filter_by_mode(terminals: pd.DataFrame, mode: str | None) -> pd.DataFrame:
    if mode is None:
        return terminals
    modes = terminals["route_types"].fillna("").str.split(",")
    return terminals[modes.map(lambda m: mode in m)].reset_index(drop=True)


def query_in_chunks(
    index: TerminalIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    chunk_size: int,
    max_km: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    distance_km = np.empty(len(lat))
    found = np.empty(len(lat), dtype=np.int64)
    for start in range(0, len(lat), chunk_size):
        stop = start + chunk_size
        distance_km[start:stop], found[start:stop] = index.query(
            lat[start:stop], lon[start:stop], max_km
        )
    return distance_km, found


def attach_matches(
    places: pd.DataFrame, index: TerminalIndex, distance_km: np.ndarray, found: np.ndarray
) -> pd.DataFrame:
    hit = found >= 0
    picked = index.terminals.iloc[np.where(hit, found, 0)]
    out = places.copy()
    for column, source in (
        ("terminal_feed_id", "feed_id"),
        ("terminal_stop_id", "stop_id"),
        ("terminal_name", "name"),
    ):
        out[column] = np.where(hit, picked[source].astype(object).to_numpy(), None)
    out["distance_km"] = distance_km
    return out


def link_places(
    places_path: Path,
    index: TerminalIndex,
    out_path: Path,
    *,
    chunk_size: int,
    max_km: float | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> tuple[int, int]:
    source = pq.ParquetFile(places_path)
    place_schema = source.schema_arrow
    for column in (lat_col, lon_col):
        if column not in place_schema.names:
            raise ValueError(f"{places_path} has no column {column!r}")
    taken = set(place_schema.names) & {f.name for f in MATCH_FIELDS}
    if taken:
        raise ValueError(f"{places_path} already has columns {sorted(taken)}")
    schema = pa.schema(list(place_schema) + MATCH_FIELDS)

    rows = matched = 0
    with pq.ParquetWriter(out_path, schema) as writer:
        for batch in source.iter_batches(batch_size=chunk_size):
            places = batch.to_pandas()
            distance_km, found = index.query(
                places[lat_col].to_numpy(dtype=np.float64, na_value=np.nan),
                places[lon_col].to_numpy(dtype=np.float64, na_value=np.nan),
                max_km,
            )
            out = attach_matches(places, index, distance_km, found)
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            rows += len(out)
            matched += int((found >= 0).sum())
    return rows, matched
