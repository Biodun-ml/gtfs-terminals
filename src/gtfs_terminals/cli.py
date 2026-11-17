import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from gtfs_terminals.dedup import dedupe_across_feeds
from gtfs_terminals.detect import Thresholds, find_feed_terminals
from gtfs_terminals.feed import FeedError
from gtfs_terminals.geocode import GeocoderUnavailable, add_admin_regions
from gtfs_terminals.nearest import TerminalIndex, filter_by_mode, link_places

log = logging.getLogger("gtfs_terminals")


def run_find(args: argparse.Namespace) -> int:
    thresholds = Thresholds(min_trips=args.min_trips, min_routes=args.min_routes)
    per_feed = [find_feed_terminals(path, thresholds, args.chunk) for path in args.feeds]
    terminals = pd.concat(per_feed, ignore_index=True)
    terminals = dedupe_across_feeds(terminals, radius_m=args.dedup_m)

    if args.geocode:
        try:
            terminals = add_admin_regions(terminals)
        except GeocoderUnavailable as exc:
            log.warning("skipping reverse geocoding: %s", exc)

    terminals.to_parquet(args.output, index=False)
    merged = sum(len(f) for f in per_feed) - len(terminals)
    log.info(
        "wrote %d terminals to %s (%d cross-feed duplicates merged)",
        len(terminals),
        args.output,
        merged,
    )
    return 0


def run_nearest(args: argparse.Namespace) -> int:
    terminals = filter_by_mode(pd.read_parquet(args.terminals), args.mode)
    if terminals.empty:
        log.error("no terminals with route type %r", args.mode)
        return 1
    started = time.perf_counter()
    rows, matched = link_places(
        args.places,
        TerminalIndex(terminals),
        args.output,
        chunk_size=args.chunk,
        max_km=args.max_km,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
    )
    log.info(
        "%d places, %d matched against %d terminals in %.1fs -> %s",
        rows,
        matched,
        len(terminals),
        time.perf_counter() - started,
        args.output,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gtfs-terminals")
    commands = parser.add_subparsers(dest="command", required=True)

    find = commands.add_parser("find", help="detect terminals in GTFS zips")
    find.add_argument("feeds", nargs="+", type=Path)
    find.add_argument("-o", "--output", type=Path, required=True)
    find.add_argument(
        "--min-trips",
        type=int,
        default=20,
        help="keep stops that start or end at least this many trips (default 20)",
    )
    find.add_argument(
        "--min-routes",
        type=int,
        default=3,
        help="or that start or end at least this many routes (default 3)",
    )
    find.add_argument(
        "--dedup-m",
        type=float,
        default=50.0,
        help="merge same-named terminals from different feeds within this radius",
    )
    find.add_argument("--chunk", type=int, default=1_000_000, help="stop_times.txt rows per chunk")
    find.add_argument(
        "--geocode", action="store_true", help="add country_code and admin1 (needs the [geo] extra)"
    )
    find.set_defaults(run=run_find)

    nearest = commands.add_parser("nearest", help="link places to their nearest terminal")
    nearest.add_argument("places", type=Path)
    nearest.add_argument("terminals", type=Path)
    nearest.add_argument("-o", "--output", type=Path, required=True)
    nearest.add_argument("--mode", help="only terminals serving this route type, e.g. ferry")
    nearest.add_argument("--max-km", type=float, help="leave places farther than this unmatched")
    nearest.add_argument("--chunk", type=int, default=200_000, help="places per chunk")
    nearest.add_argument("--lat-col", default="lat")
    nearest.add_argument("--lon-col", default="lon")
    nearest.set_defaults(run=run_nearest)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except (FeedError, ValueError, FileNotFoundError) as exc:
        log.error("error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
