import zipfile

import pandas as pd
import pytest
from conftest import write_feed

from gtfs_terminals.detect import Thresholds, find_feed_terminals, trip_endpoints
from gtfs_terminals.feed import FeedError

EVERYTHING = Thresholds(min_trips=0, min_routes=0)


def counts(terminals: pd.DataFrame) -> dict[str, int]:
    return dict(zip(terminals["stop_id"], terminals["endpoint_trip_count"], strict=True))


@pytest.mark.parametrize("chunksize", [1, 3, 7, 1_000])
def test_endpoint_counts_do_not_depend_on_chunking(metro_feed, chunksize):
    terminals = find_feed_terminals(metro_feed, EVERYTHING, chunksize=chunksize)
    assert counts(terminals) == {
        "CEN": 6,
        "HAR": 6,
        "LOOP": 4,
        "PIER": 5,
        "ISL": 5,
        "MID": 1,
        "QH": 1,
    }


def test_platforms_roll_up_to_parent_station(metro_feed):
    terminals = find_feed_terminals(metro_feed, EVERYTHING).set_index("stop_id")
    assert "CEN-1" not in terminals.index
    assert "CEN-2" not in terminals.index
    assert terminals.loc["CEN", "name"] == "Central Station"
    assert terminals.loc["CEN", "route_types"] == "rail"


def test_loop_trip_counts_once_at_its_terminus(metro_feed):
    terminals = find_feed_terminals(metro_feed, EVERYTHING).set_index("stop_id")
    assert terminals.loc["LOOP", "endpoint_trip_count"] == 4
    assert terminals.loc["LOOP", "distinct_routes"] == 1


def test_mid_route_stops_are_not_endpoints(metro_feed):
    terminals = find_feed_terminals(metro_feed, EVERYTHING)
    assert not {"A", "B"} & set(terminals["stop_id"])


def test_ferry_route_type_is_labelled(metro_feed):
    terminals = find_feed_terminals(metro_feed, EVERYTHING).set_index("stop_id")
    assert terminals.loc["PIER", "route_types"] == "ferry"
    assert terminals.loc["ISL", "route_types"] == "ferry"


def test_thresholds_drop_low_service_stops(metro_feed):
    terminals = find_feed_terminals(metro_feed, Thresholds(min_trips=2, min_routes=99))
    assert set(terminals["stop_id"]) == {"CEN", "HAR", "LOOP", "PIER", "ISL"}


def test_route_threshold_alone_can_qualify_a_stop(regional_feed):
    terminals = find_feed_terminals(regional_feed, Thresholds(min_trips=99, min_routes=2))
    assert list(terminals["stop_id"]) == ["RX"]


def test_feed_without_hierarchy_columns_and_with_bom(regional_feed):
    terminals = find_feed_terminals(regional_feed, EVERYTHING).set_index("stop_id")
    assert counts(terminals.reset_index()) == {"RX": 4, "RC": 3, "RM": 1}
    assert terminals.loc["RC", "name"] == "CENTRAL STATION."
    assert terminals.loc["RC", "lat"] == pytest.approx(51.50015)
    assert terminals.loc["RX", "distinct_routes"] == 2


def test_station_without_coordinates_uses_platform_centroid(tmp_path):
    feed = write_feed(
        tmp_path / "f.zip",
        {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
            "S,Station,,,1,\nS1,P1,10.0,20.0,0,S\nS2,P2,10.2,20.2,0,S\nT,Other,11,21,0,\n",
            "routes.txt": "route_id,route_type\nR,3\n",
            "trips.txt": "route_id,service_id,trip_id\nR,WK,t1\n",
            "stop_times.txt": "trip_id,stop_id,stop_sequence\nt1,S1,1\nt1,T,2\n",
        },
    )
    terminals = find_feed_terminals(feed, EVERYTHING).set_index("stop_id")
    assert terminals.loc["S", ["lat", "lon"]].tolist() == pytest.approx([10.1, 20.1])


def test_missing_stop_times_is_a_feed_error(tmp_path):
    feed = tmp_path / "broken.zip"
    with zipfile.ZipFile(feed, "w") as zf:
        zf.writestr("stops.txt", "stop_id,stop_lat,stop_lon\nA,1,1\n")
        zf.writestr("trips.txt", "route_id,trip_id\nR,t\n")
        zf.writestr("routes.txt", "route_id,route_type\nR,3\n")
    with pytest.raises(FeedError, match="stop_times.txt"):
        find_feed_terminals(feed, EVERYTHING)


def test_trip_endpoints_handles_unsorted_sequences_across_chunks():
    chunks = [
        pd.DataFrame(
            {"trip_id": ["t", "t"], "stop_id": ["mid", "last"], "stop_sequence": ["5", "9"]}
        ),
        pd.DataFrame({"trip_id": ["t"], "stop_id": ["first"], "stop_sequence": ["1"]}),
    ]
    ends = trip_endpoints(iter(chunks))
    assert sorted(ends["stop_id"]) == ["first", "last"]
