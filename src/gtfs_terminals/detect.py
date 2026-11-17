import logging
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gtfs_terminals.feed import iter_table, read_table, require_columns
from gtfs_terminals.route_types import label_route_types

log = logging.getLogger(__name__)

TERMINAL_COLUMNS = [
    "feed_id",
    "stop_id",
    "name",
    "lat",
    "lon",
    "route_types",
    "endpoint_trip_count",
    "distinct_routes",
]

STATION = 1
STOP_TIME_COLUMNS = {"trip_id", "stop_id", "stop_sequence"}


@dataclass(frozen=True)
class Thresholds:
    min_trips: int = 20
    min_routes: int = 3


def load_stops(zf: zipfile.ZipFile) -> pd.DataFrame:
    wanted = {"stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"}
    stops = read_table(zf, "stops.txt", wanted)
    require_columns(stops, "stops.txt", {"stop_id", "stop_lat", "stop_lon"})
    for column in ("stop_name", "parent_station", "location_type"):
        if column not in stops.columns:
            stops[column] = ""
    stops["location_type"] = (
        pd.to_numeric(stops["location_type"], errors="coerce").fillna(0).astype("int64")
    )
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    return stops.drop_duplicates("stop_id", ignore_index=True)


def station_of(stops: pd.DataFrame) -> pd.Series:
    ids = stops["stop_id"]
    declared = stops["parent_station"]
    parent = declared.where(declared.isin(ids) & (declared != ""), ids)
    parent = pd.Series(parent.to_numpy(), index=ids.to_numpy())
    # Boarding areas point at platforms, which point at stations; follow at most three hops
    # so a malformed cycle cannot loop forever.
    station = parent
    for _ in range(3):
        hop = pd.Series(parent.reindex(station.to_numpy()).to_numpy(), index=station.index)
        if hop.equals(station):
            break
        station = hop
    return station


def _first_or_last(parts: list[pd.DataFrame], keep: str) -> pd.DataFrame:
    # A trip can straddle a chunk boundary, so the per-chunk picks get reduced once more.
    combined = pd.concat(parts, ignore_index=True).sort_values("stop_sequence", kind="stable")
    return combined.drop_duplicates("trip_id", keep=keep)


def trip_endpoints(stop_times: Iterable[pd.DataFrame]) -> pd.DataFrame:
    firsts: list[pd.DataFrame] = []
    lasts: list[pd.DataFrame] = []
    for chunk in stop_times:
        chunk = chunk.assign(stop_sequence=pd.to_numeric(chunk["stop_sequence"], errors="coerce"))
        chunk = chunk.dropna(subset=["stop_sequence"]).sort_values("stop_sequence", kind="stable")
        firsts.append(chunk.drop_duplicates("trip_id", keep="first"))
        lasts.append(chunk.drop_duplicates("trip_id", keep="last"))
    if not firsts:
        return pd.DataFrame({"trip_id": pd.Series(dtype=str), "stop_id": pd.Series(dtype=str)})
    ends = pd.concat([_first_or_last(firsts, "first"), _first_or_last(lasts, "last")])
    return ends[["trip_id", "stop_id"]].drop_duplicates(ignore_index=True)


def endpoint_stats(
    endpoints: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame, station: pd.Series
) -> pd.DataFrame:
    events = endpoints.assign(station_id=endpoints["stop_id"].map(station))
    events = events.dropna(subset=["station_id"])
    events = events.merge(trips.drop_duplicates("trip_id"), on="trip_id", how="left")
    events = events.merge(routes.drop_duplicates("route_id"), on="route_id", how="left")
    events["mode"] = label_route_types(events["route_type"])
    grouped = events.groupby("station_id")
    return pd.DataFrame(
        {
            "endpoint_trip_count": grouped["trip_id"].nunique(),
            "distinct_routes": grouped["route_id"].nunique(),
            "route_types": grouped["mode"].agg(lambda modes: ",".join(sorted(set(modes)))),
        }
    )


def _stop_time_chunks(zf: zipfile.ZipFile, chunksize: int) -> Iterator[pd.DataFrame]:
    for chunk in iter_table(zf, "stop_times.txt", STOP_TIME_COLUMNS, chunksize):
        require_columns(chunk, "stop_times.txt", STOP_TIME_COLUMNS)
        yield chunk


def _station_coords(stops: pd.DataFrame, station: pd.Series) -> pd.DataFrame:
    # Parent stations may omit coordinates; fall back to the centroid of their platforms.
    own = stops.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]]
    children = (
        stops.assign(station_id=stops["stop_id"].map(station))
        .groupby("station_id")[["stop_lat", "stop_lon"]]
        .mean()
    )
    own["stop_lat"] = own["stop_lat"].fillna(children["stop_lat"])
    own["stop_lon"] = own["stop_lon"].fillna(children["stop_lon"])
    return own


def select_terminals(
    stops: pd.DataFrame, stats: pd.DataFrame, thresholds: Thresholds, feed_id: str
) -> pd.DataFrame:
    stations = stops.loc[stops["location_type"] == STATION, "stop_id"]
    candidates = stats.reindex(pd.Index(stations).union(stats.index))
    candidates = candidates.fillna(
        {"endpoint_trip_count": 0, "distinct_routes": 0, "route_types": ""}
    )
    keep = (candidates["endpoint_trip_count"] >= thresholds.min_trips) | (
        candidates["distinct_routes"] >= thresholds.min_routes
    )
    kept = candidates[keep]

    coords = _station_coords(stops, station_of(stops)).reindex(kept.index)
    terminals = pd.DataFrame(
        {
            "feed_id": feed_id,
            "stop_id": kept.index.to_numpy(),
            "name": coords["stop_name"].to_numpy(),
            "lat": coords["stop_lat"].to_numpy(),
            "lon": coords["stop_lon"].to_numpy(),
            "route_types": kept["route_types"].to_numpy(),
            "endpoint_trip_count": kept["endpoint_trip_count"].astype("int64").to_numpy(),
            "distinct_routes": kept["distinct_routes"].astype("int64").to_numpy(),
        }
    )
    no_coords = terminals["lat"].isna() | terminals["lon"].isna()
    if no_coords.any():
        log.warning("%s: dropping %d terminals without coordinates", feed_id, no_coords.sum())
    return terminals[~no_coords].reset_index(drop=True)


def find_feed_terminals(
    path: Path, thresholds: Thresholds, chunksize: int = 1_000_000, feed_id: str | None = None
) -> pd.DataFrame:
    feed_id = feed_id or path.stem
    with zipfile.ZipFile(path) as zf:
        stops = load_stops(zf)
        trips = read_table(zf, "trips.txt", {"trip_id", "route_id"})
        require_columns(trips, "trips.txt", {"trip_id", "route_id"})
        routes = read_table(zf, "routes.txt", {"route_id", "route_type"})
        require_columns(routes, "routes.txt", {"route_id", "route_type"})
        endpoints = trip_endpoints(_stop_time_chunks(zf, chunksize))

    stats = endpoint_stats(endpoints, trips, routes, station_of(stops))
    terminals = select_terminals(stops, stats, thresholds, feed_id)
    log.info(
        "%s: %d stops, %d trips, %d terminals", feed_id, len(stops), len(trips), len(terminals)
    )
    return terminals
