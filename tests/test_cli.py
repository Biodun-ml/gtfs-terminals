import sys

import pandas as pd

from gtfs_terminals.cli import main


def test_find_then_nearest(tmp_path, metro_feed, regional_feed):
    terminals_path = tmp_path / "terminals.parquet"
    assert (
        main(
            [
                "find",
                str(metro_feed),
                str(regional_feed),
                "-o",
                str(terminals_path),
                "--min-trips",
                "2",
            ]
        )
        == 0
    )
    terminals = pd.read_parquet(terminals_path)
    assert "CEN" in set(terminals["stop_id"])

    places = pd.DataFrame(
        {"name": ["near pier", "nowhere"], "y": [51.506, -40.0], "x": [-0.05, 100.0]}
    )
    places.to_parquet(tmp_path / "places.parquet", index=False)
    out_path = tmp_path / "out.parquet"
    code = main(
        [
            "nearest",
            str(tmp_path / "places.parquet"),
            str(terminals_path),
            "-o",
            str(out_path),
            "--mode",
            "ferry",
            "--max-km",
            "50",
            "--lat-col",
            "y",
            "--lon-col",
            "x",
        ]
    )
    assert code == 0
    out = pd.read_parquet(out_path)
    assert out["terminal_stop_id"].tolist()[0] == "PIER"
    assert pd.isna(out["terminal_stop_id"].tolist()[1])


def test_geocode_is_skipped_without_the_package(tmp_path, metro_feed, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "reverse_geocoder", None)
    out = tmp_path / "t.parquet"
    assert main(["find", str(metro_feed), "-o", str(out), "--geocode"]) == 0
    assert "skipping reverse geocoding" in caplog.text
    assert "country_code" not in pd.read_parquet(out).columns


def test_unknown_mode_fails_cleanly(tmp_path, metro_feed):
    terminals_path = tmp_path / "terminals.parquet"
    main(["find", str(metro_feed), "-o", str(terminals_path), "--min-trips", "1"])
    pd.DataFrame({"lat": [0.0], "lon": [0.0]}).to_parquet(tmp_path / "p.parquet")
    code = main(
        [
            "nearest",
            str(tmp_path / "p.parquet"),
            str(terminals_path),
            "-o",
            str(tmp_path / "o.parquet"),
            "--mode",
            "funicular",
        ]
    )
    assert code == 1
