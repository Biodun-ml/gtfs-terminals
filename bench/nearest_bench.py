"""Time nearest-terminal lookups: chunked KD-tree vs a pandas cross join.

    python bench/nearest_bench.py                      # full suite, prints a markdown table
    python bench/nearest_bench.py --case kdtree --places 1000000 --terminals 20000

Each case runs in its own subprocess so peak RSS belongs to that case alone.
"""

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gtfs_terminals.geo import haversine_km
from gtfs_terminals.nearest import TerminalIndex, link_places, query_in_chunks

SUITE = [
    ("naive", 250, 20_000, 0),
    ("naive", 500, 20_000, 0),
    ("naive", 1_000, 20_000, 0),
    ("kdtree", 1_000, 20_000, 200_000),
    ("kdtree", 100_000, 20_000, 200_000),
    ("kdtree", 1_000_000, 20_000, 200_000),
    ("kdtree", 1_000_000, 20_000, 50_000),
    ("kdtree", 1_000_000, 20_000, 1_000_000),
    ("kdtree", 5_000_000, 20_000, 200_000),
    ("parquet", 1_000_000, 20_000, 200_000),
]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def random_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    return lat, rng.uniform(-180, 180, n)


def make_terminals(rng: np.random.Generator, n: int) -> pd.DataFrame:
    lat, lon = random_points(rng, n)
    return pd.DataFrame(
        {
            "feed_id": "bench",
            "stop_id": np.arange(n).astype(str),
            "name": "t",
            "lat": lat,
            "lon": lon,
        }
    )


def run_naive(places: pd.DataFrame, terminals: pd.DataFrame, _chunk: int) -> None:
    pairs = places.merge(terminals[["stop_id", "lat", "lon"]], how="cross", suffixes=("", "_t"))
    pairs["km"] = haversine_km(pairs["lat"], pairs["lon"], pairs["lat_t"], pairs["lon_t"])
    pairs.loc[pairs.groupby("place_id")["km"].idxmin()]


def run_kdtree(places: pd.DataFrame, terminals: pd.DataFrame, chunk: int) -> None:
    index = TerminalIndex(terminals)
    query_in_chunks(index, places["lat"].to_numpy(), places["lon"].to_numpy(), chunk)


def run_parquet(places: Path, terminals: pd.DataFrame, chunk: int) -> None:
    link_places(places, TerminalIndex(terminals), places.with_name("out.parquet"), chunk_size=chunk)


RUNNERS = {"naive": run_naive, "kdtree": run_kdtree, "parquet": run_parquet}


def run_case(case: str, n_places: int, n_terminals: int, chunk: int, tmp: Path) -> dict:
    rng = np.random.default_rng(0)
    lat, lon = random_points(rng, n_places)
    places = pd.DataFrame({"place_id": np.arange(n_places), "lat": lat, "lon": lon})
    terminals = make_terminals(rng, n_terminals)
    if case == "parquet":
        # The CLI never holds all places in memory, so neither should this case.
        path = tmp / "places.parquet"
        places.to_parquet(path, index=False, row_group_size=chunk)
        places = path
    baseline = peak_rss_mb()

    started = time.perf_counter()
    RUNNERS[case](places, terminals, chunk)
    seconds = time.perf_counter() - started
    peak = peak_rss_mb()
    return {
        "case": case,
        "places": n_places,
        "terminals": n_terminals,
        "chunk": chunk,
        "pairs": n_places * n_terminals,
        "seconds": round(seconds, 2),
        "peak_rss_mb": round(peak),
        "extra_rss_mb": round(peak - baseline),
    }


def run_suite() -> None:
    print("| method | places | terminals | chunk | seconds | peak RSS MB | above inputs MB |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for case, n_places, n_terminals, chunk in SUITE:
        cmd = [sys.executable, __file__, "--case", case, "--places", str(n_places),
               "--terminals", str(n_terminals), "--chunk", str(chunk)]  # fmt: skip
        r = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
        chunk_label = f"{r['chunk']:,}" if r["chunk"] else "-"
        print(
            f"| {r['case']} | {r['places']:,} | {r['terminals']:,} | {chunk_label} "
            f"| {r['seconds']} | {r['peak_rss_mb']:,} | {r['extra_rss_mb']:,} |",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(RUNNERS))
    parser.add_argument("--places", type=int, default=1_000_000)
    parser.add_argument("--terminals", type=int, default=20_000)
    parser.add_argument("--chunk", type=int, default=200_000)
    args = parser.parse_args()
    if args.case is None:
        run_suite()
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(args.case, args.places, args.terminals, args.chunk, Path(tmp))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
