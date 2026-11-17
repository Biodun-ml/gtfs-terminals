import numpy as np
import pandas as pd
import pytest

from gtfs_terminals.geo import haversine_km
from gtfs_terminals.nearest import TerminalIndex, filter_by_mode, link_places, query_in_chunks


def random_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    lon = rng.uniform(-180, 180, n)
    return lat, lon


def make_terminals(lat: np.ndarray, lon: np.ndarray, modes: list[str] | None = None):
    n = len(lat)
    return pd.DataFrame(
        {
            "feed_id": "f",
            "stop_id": [f"t{i}" for i in range(n)],
            "name": [f"Terminal {i}" for i in range(n)],
            "lat": lat,
            "lon": lon,
            "route_types": modes or ["bus"] * n,
            "endpoint_trip_count": 10,
            "distinct_routes": 1,
        }
    )


def brute_force(place_lat, place_lon, term_lat, term_lon):
    km = haversine_km(place_lat[:, None], place_lon[:, None], term_lat[None, :], term_lon[None, :])
    return km.min(axis=1), km.argmin(axis=1)


def test_matches_brute_force_haversine():
    rng = np.random.default_rng(7)
    t_lat, t_lon = random_points(rng, 400)
    p_lat, p_lon = random_points(rng, 3_000)

    km, idx = TerminalIndex(make_terminals(t_lat, t_lon)).query(p_lat, p_lon)
    want_km, want_idx = brute_force(p_lat, p_lon, t_lat, t_lon)

    np.testing.assert_allclose(km, want_km, rtol=1e-9, atol=1e-6)
    np.testing.assert_array_equal(idx, want_idx)


def test_antimeridian_neighbour_is_found():
    index = TerminalIndex(make_terminals(np.array([0.0, 0.0]), np.array([179.9, 100.0])))
    km, idx = index.query(np.array([0.0]), np.array([-179.9]))
    assert idx[0] == 0
    assert km[0] == pytest.approx(22.24, abs=0.01)


@pytest.mark.parametrize("chunk_size", [1, 17, 1_000, 10_000])
def test_chunk_size_does_not_change_results(chunk_size):
    rng = np.random.default_rng(11)
    index = TerminalIndex(make_terminals(*random_points(rng, 200)))
    p_lat, p_lon = random_points(rng, 2_000)

    whole_km, whole_idx = index.query(p_lat, p_lon, max_km=500)
    km, idx = query_in_chunks(index, p_lat, p_lon, chunk_size, max_km=500)

    np.testing.assert_array_equal(idx, whole_idx)
    np.testing.assert_array_equal(km, whole_km)


def test_max_km_leaves_far_places_unmatched():
    index = TerminalIndex(make_terminals(np.array([0.0]), np.array([0.0])))
    km, idx = index.query(np.array([0.0, 0.0, np.nan]), np.array([0.1, 5.0, 0.0]), max_km=50)
    assert idx.tolist() == [0, -1, -1]
    assert km[0] == pytest.approx(11.12, abs=0.01)
    assert np.isnan(km[1:]).all()


def test_filter_by_mode():
    terminals = make_terminals(np.zeros(3), np.arange(3.0), modes=["bus", "bus,ferry", "rail"])
    assert filter_by_mode(terminals, "ferry")["stop_id"].tolist() == ["t1"]
    assert len(filter_by_mode(terminals, None)) == 3


def test_link_places_streams_parquet(tmp_path):
    rng = np.random.default_rng(3)
    t_lat, t_lon = random_points(rng, 50)
    p_lat, p_lon = random_points(rng, 1_001)
    places = pd.DataFrame({"place_id": np.arange(1_001), "lat": p_lat, "lon": p_lon})
    places.to_parquet(tmp_path / "places.parquet", index=False)

    index = TerminalIndex(make_terminals(t_lat, t_lon))
    rows, matched = link_places(
        tmp_path / "places.parquet", index, tmp_path / "out.parquet", chunk_size=100, max_km=800
    )
    out = pd.read_parquet(tmp_path / "out.parquet")

    want_km, want_idx = brute_force(p_lat, p_lon, t_lat, t_lon)
    within = want_km <= 800
    assert rows == 1_001
    assert matched == within.sum()
    assert out["place_id"].tolist() == list(range(1_001))
    assert out.loc[within, "terminal_stop_id"].tolist() == [f"t{i}" for i in want_idx[within]]
    assert out.loc[~within, "terminal_stop_id"].isna().all()
