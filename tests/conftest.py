import zipfile
from pathlib import Path

import pytest


def write_feed(path: Path, tables: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, body in tables.items():
            zf.writestr(name, body)
    return path


def stop_times_for(trips: list[tuple[str, list[str]]]) -> str:
    lines = ["trip_id,arrival_time,departure_time,stop_id,stop_sequence"]
    for trip_id, stop_ids in trips:
        for seq, stop_id in enumerate(stop_ids, start=1):
            lines.append(f"{trip_id},08:00:00,08:00:00,{stop_id},{seq * 10}")
    return "\n".join(lines) + "\n"


def trips_for(trips: list[tuple[str, str]]) -> str:
    return "route_id,service_id,trip_id\n" + "".join(f"{r},WK,{t}\n" for t, r in trips)


@pytest.fixture
def metro_feed(tmp_path: Path) -> Path:
    # Central has two platforms under one parent station. R1 runs Central <-> Harbour,
    # departing from alternate platforms. L1 is a loop starting and ending at Loop Terminus.
    # F1 is a ferry between Pier 1 and Island Wharf. Quiet Halt ends a single trip.
    stops = (
        "stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station\n"
        "CEN,Central Station,51.5000,-0.1000,1,\n"
        "CEN-1,Central Station Platform 1,51.5001,-0.1001,0,CEN\n"
        "CEN-2,Central Station Platform 2,51.4999,-0.0999,0,CEN\n"
        "MID,Midtown,51.5100,-0.1200,0,\n"
        "HAR,Harbour,51.5200,-0.1400,0,\n"
        "LOOP,Loop Terminus,51.4800,-0.0800,0,\n"
        "A,Ash Road,51.4810,-0.0810,0,\n"
        "B,Birch Road,51.4820,-0.0790,0,\n"
        "PIER,Pier 1,51.5050,-0.0500,0,\n"
        "ISL,Island Wharf,51.4500,0.0500,0,\n"
        "QH,Quiet Halt,51.5300,-0.1500,0,\n"
    )
    routes = "route_id,route_short_name,route_type\nR1,R1,2\nL1,L1,3\nF1,F1,4\nQ1,Q1,3\n"
    trip_rows: list[tuple[str, str]] = []
    trip_stops: list[tuple[str, list[str]]] = []
    for n in range(6):
        platform = "CEN-1" if n % 2 == 0 else "CEN-2"
        path = [platform, "MID", "HAR"] if n < 3 else ["HAR", "MID", platform]
        trip_rows.append((f"r1-{n}", "R1"))
        trip_stops.append((f"r1-{n}", path))
    for n in range(4):
        trip_rows.append((f"l1-{n}", "L1"))
        trip_stops.append((f"l1-{n}", ["LOOP", "A", "B", "LOOP"]))
    for n in range(5):
        trip_rows.append((f"f1-{n}", "F1"))
        trip_stops.append((f"f1-{n}", ["PIER", "ISL"] if n % 2 == 0 else ["ISL", "PIER"]))
    trip_rows.append(("q1-0", "Q1"))
    trip_stops.append(("q1-0", ["MID", "QH"]))

    return write_feed(
        tmp_path / "metro.zip",
        {
            "stops.txt": stops,
            "routes.txt": routes,
            "trips.txt": trips_for(trip_rows),
            "stop_times.txt": stop_times_for(trip_stops),
        },
    )


@pytest.fixture
def regional_feed(tmp_path: Path) -> Path:
    # No location_type or parent_station columns, a BOM, padded headers and values,
    # and the files nested in a folder. Its "CENTRAL STATION." is ~20 m from metro's.
    stops = (
        "﻿stop_id , stop_name , stop_lat , stop_lon\n"
        "RC , CENTRAL STATION. , 51.50015 , -0.10020\n"
        "RX , Crossroads , 51.6000 , -0.2000\n"
        "RM , Central Station , 51.7000 , -0.3000\n"
    )
    routes = "﻿route_id,route_type\nX9,3\nX10,3\n"
    trip_rows = [(f"x9-{n}", "X9") for n in range(3)] + [("x10-0", "X10")]
    trip_stops = [(t, ["RC", "RX"]) for t, _ in trip_rows[:3]] + [("x10-0", ["RX", "RM"])]
    return write_feed(
        tmp_path / "regional.zip",
        {
            "regional/stops.txt": stops,
            "regional/routes.txt": routes,
            "regional/trips.txt": trips_for(trip_rows),
            "regional/stop_times.txt": stop_times_for(trip_stops),
        },
    )
