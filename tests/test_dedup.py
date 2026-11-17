import pandas as pd

from gtfs_terminals.dedup import dedupe_across_feeds, normalise_name
from gtfs_terminals.detect import Thresholds, find_feed_terminals

EVERYTHING = Thresholds(min_trips=0, min_routes=0)


def test_normalise_name():
    assert normalise_name("  CENTRAL  Station. ") == "central station"
    assert normalise_name("Estación Atocha") == "estacion atocha"
    assert normalise_name("東京駅") == "東京駅"


def test_same_terminal_in_two_feeds_is_merged(metro_feed, regional_feed):
    terminals = pd.concat(
        [
            find_feed_terminals(metro_feed, EVERYTHING),
            find_feed_terminals(regional_feed, EVERYTHING),
        ],
        ignore_index=True,
    )
    merged = dedupe_across_feeds(terminals, radius_m=50)

    assert len(merged) == len(terminals) - 1
    central = merged[merged["source_feeds"] == "metro,regional"].iloc[0]
    assert central["stop_id"] == "CEN"
    assert central["feed_id"] == "metro"
    assert central["endpoint_trip_count"] == 6 + 3
    assert central["distinct_routes"] == 2
    assert central["route_types"] == "bus,rail"

    far_namesake = merged[merged["stop_id"] == "RM"]
    assert len(far_namesake) == 1


def test_nearby_terminals_with_different_names_stay_apart():
    terminals = pd.DataFrame(
        {
            "feed_id": ["a", "b"],
            "stop_id": ["1", "2"],
            "name": ["North Pier", "South Pier"],
            "lat": [10.0, 10.0001],
            "lon": [20.0, 20.0],
            "route_types": ["ferry", "ferry"],
            "endpoint_trip_count": [5, 5],
            "distinct_routes": [1, 1],
        }
    )
    assert len(dedupe_across_feeds(terminals)) == 2


def test_same_feed_namesakes_are_not_merged():
    terminals = pd.DataFrame(
        {
            "feed_id": ["a", "a"],
            "stop_id": ["1", "2"],
            "name": ["Mall", "Mall"],
            "lat": [10.0, 10.0001],
            "lon": [20.0, 20.0],
            "route_types": ["bus", "bus"],
            "endpoint_trip_count": [5, 7],
            "distinct_routes": [1, 1],
        }
    )
    assert len(dedupe_across_feeds(terminals)) == 2
