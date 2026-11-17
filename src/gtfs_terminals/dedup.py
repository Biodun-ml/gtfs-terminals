import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from gtfs_terminals.geo import km_to_chord, unit_xyz


def normalise_name(name: str) -> str:
    # Strip accents but keep non-Latin letters; an ASCII-only fold would turn every
    # Japanese or Arabic name into "" and make them all match each other.
    decomposed = unicodedata.normalize("NFKD", name)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[\W_]+", " ", no_marks.casefold()).split())


def duplicate_groups(terminals: pd.DataFrame, radius_m: float) -> np.ndarray:
    n = len(terminals)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(unit_xyz(terminals["lat"].to_numpy(), terminals["lon"].to_numpy()))
    pairs = tree.query_pairs(float(km_to_chord(radius_m / 1000.0)), output_type="ndarray")

    names = terminals["name"].fillna("").map(normalise_name).to_numpy()
    feeds = terminals["feed_id"].to_numpy()
    i, j = pairs[:, 0], pairs[:, 1]
    same = (names[i] == names[j]) & (feeds[i] != feeds[j]) & (names[i] != "")
    i, j = i[same], j[same]

    graph = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def _join_sorted(values: pd.Series) -> str:
    parts = {part for value in values for part in value.split(",") if part}
    return ",".join(sorted(parts))


def dedupe_across_feeds(terminals: pd.DataFrame, radius_m: float = 50.0) -> pd.DataFrame:
    if terminals.empty:
        return terminals.assign(source_feeds=pd.Series(dtype=str))
    labels = duplicate_groups(terminals, radius_m)
    # The busiest member keeps its id, name and position; trips and routes are summed
    # because each feed contributes different services at the shared terminal.
    ranked = terminals.assign(_group=labels).sort_values(
        ["_group", "endpoint_trip_count"], ascending=[True, False], kind="stable"
    )
    grouped = ranked.groupby("_group", sort=False)
    merged = grouped.first()
    merged["endpoint_trip_count"] = grouped["endpoint_trip_count"].sum()
    merged["distinct_routes"] = grouped["distinct_routes"].sum()
    merged["route_types"] = grouped["route_types"].agg(_join_sorted)
    merged["source_feeds"] = grouped["feed_id"].agg(_join_sorted)
    return merged.sort_values("endpoint_trip_count", ascending=False).reset_index(drop=True)
