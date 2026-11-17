# gtfs-terminals

Finds the terminals (stations, bus termini, ferry piers) in GTFS feeds, then links any number of
places to their nearest terminal. I wrote it to attach millions of points of interest to the
closest bus, rail or ferry terminal on a 16 GB laptop, where a pandas cross join was not an option.

## Install

```
pip install -e .            # pandas, pyarrow, numpy, scipy
pip install -e ".[geo]"     # adds reverse_geocoder for --geocode
pip install -e ".[dev]"     # pytest, ruff
```

Python 3.10+.

## Usage

```
gtfs-terminals find FEED.zip [FEED.zip ...] -o terminals.parquet
    [--min-trips 20] [--min-routes 3] [--dedup-m 50] [--chunk 1000000] [--geocode]

gtfs-terminals nearest places.parquet terminals.parquet -o out.parquet
    [--mode ferry] [--max-km 50] [--chunk 200000] [--lat-col lat] [--lon-col lon]
```

Example with three public feeds: BART (`https://www.bart.gov/dev/schedules/google_transit.zip`),
Washington State Ferries (`https://gtfs.sound.obaweb.org/prod/95_gtfs.zip`, saved as `wsf.zip`)
and MTA Staten Island buses (`https://rrgtfsfeeds.s3.amazonaws.com/gtfs_si.zip`). Outputs below
are from feeds downloaded in November 2026.

```
$ gtfs-terminals find bart.zip wsf.zip mta_staten_island_bus.zip -o terminals.parquet --geocode
bart: 291 stops, 6377 trips, 15 terminals
wsf: 19 stops, 28235 trips, 19 terminals
mta_staten_island_bus: 1965 stops, 16524 trips, 134 terminals
wrote 168 terminals to terminals.parquet (0 cross-feed duplicates merged)
```

```
feed_id stop_id          name       lat         lon route_types  endpoint_trip_count  distinct_routes source_feeds country_code     admin1
    wsf       9    Fauntleroy 47.523178 -122.393113       ferry                 5817                4          wsf           US Washington
    wsf      22 Vashon Island 47.508431 -122.464058       ferry                 5501                4          wsf           US Washington
    wsf      20    Southworth 47.511925 -122.499543       ferry                 4806                4          wsf           US Washington
    wsf       7       Seattle 47.602597 -122.337635       ferry                 3987                4          wsf           US Washington
    wsf       5       Clinton 47.974564 -122.352037       ferry                 3836                2          wsf           US Washington
```

Linking the 172,021 places in GeoNames `cities1000` to the nearest ferry terminal within 50 km
(the places file only needs `lat`/`lon` columns; every other column is carried through):

```python
import pandas as pd

cols = {0: "geonameid", 1: "name", 4: "lat", 5: "lon", 8: "country_code"}
places = pd.read_csv("cities1000.txt", sep="\t", header=None, usecols=list(cols), quoting=3)
places.rename(columns=cols).to_parquet("places.parquet", index=False)
```

```
$ gtfs-terminals nearest places.parquet terminals.parquet -o places_ferry.parquet --mode ferry --max-km 50
172021 places, 274 matched against 19 terminals in 0.3s -> places_ferry.parquet
```

```
 geonameid          name   lat     lon country_code terminal_feed_id terminal_stop_id terminal_name  distance_km
   5811567    Southworth 47.51 -122.50           US              wsf               20    Southworth         0.17
   5795104 Friday Harbor 48.53 -123.02           US              wsf               10 Friday Harbor         0.20
   5799815      Kingston 47.80 -122.50           US              wsf               12      Kingston         0.27
```

Places with no terminal within `--max-km`, or with missing coordinates, get nulls in the
`terminal_*` columns and `distance_km`.

## How terminal detection works

1. **Read.** `stops.txt`, `trips.txt` and `routes.txt` are read whole; `stop_times.txt` is
   streamed in chunks (`--chunk`) because it is usually 10 to 100 times larger than the rest of
   the feed combined. Only `trip_id`, `stop_id` and `stop_sequence` are kept from it.
2. **Trip endpoints.** For each chunk, the lowest and highest `stop_sequence` row per trip is
   kept, then those picks are reduced again once all chunks are read, so trips split across chunk
   boundaries or written out of order still resolve correctly. Memory is roughly one row per trip.
3. **Roll up to stations.** Every stop is mapped to its `parent_station` (following
   boarding area -> platform -> station), so a station whose trips start on twelve different
   platforms is counted once. Candidates are all parent stations (`location_type=1`) plus every
   stop that starts or ends a trip.
4. **Filter.** A candidate is kept if it is an endpoint of at least `--min-trips` trips or of at
   least `--min-routes` distinct routes. A loop trip that starts and ends at the same place counts
   once. `route_types` lists the modes of the trips that end there (`bus`, `rail`, `subway`,
   `ferry`, ...; extended codes like 700 or 1200 are grouped).
5. **Cross-feed dedup.** Terminals from different feeds within `--dedup-m` metres whose names
   match after normalisation (case, accents, punctuation) are merged. The busiest one keeps its
   id and position, trip and route counts are summed, and `source_feeds` records where it came from.

Schema variations that are handled: BOMs and padded header names, padded values, missing
`location_type` / `parent_station` columns, parent stations with no coordinates (the platform
centroid is used), and feeds zipped inside a top-level folder.

## Nearest-terminal lookup

Coordinates are converted to 3D points on the unit sphere and indexed with `scipy.spatial.cKDTree`.
Euclidean (chord) distance between unit vectors grows monotonically with great-circle distance, so
the nearest point in 3D is the nearest terminal on the globe, with no distortion near the poles and
no seam at the antimeridian. The chord is converted back to kilometres afterwards, and `--max-km`
is passed to the tree as a chord bound so far-away places are rejected inside the search.

Places are read from Parquet one record batch at a time and written out incrementally, so memory
depends on `--chunk` and the number of terminals, not on the number of places.

## Benchmark

`python bench/nearest_bench.py` runs each case in a fresh process with uniformly random points on
the sphere. Measured on a 2 vCPU Xeon @ 2.1 GHz with 7 GB RAM (smaller than the 16 GB laptop
this was built for). Peak RSS includes about 170 MB for Python, pandas and scipy; "above inputs"
is the peak minus the RSS after the input arrays were built.

| method | places | terminals | chunk | seconds | peak RSS MB | above inputs MB |
|---|---:|---:|---:|---:|---:|---:|
| pandas cross join | 250 | 20,000 | - | 0.90 | 745 | 567 |
| pandas cross join | 500 | 20,000 | - | 2.03 | 1,305 | 1,127 |
| pandas cross join | 1,000 | 20,000 | - | 4.62 | 2,427 | 2,248 |
| KD-tree | 1,000 | 20,000 | 200,000 | 0.01 | 180 | 1 |
| KD-tree | 100,000 | 20,000 | 200,000 | 0.10 | 195 | 13 |
| KD-tree | 1,000,000 | 20,000 | 50,000 | 0.87 | 239 | 17 |
| KD-tree | 1,000,000 | 20,000 | 200,000 | 0.66 | 254 | 33 |
| KD-tree | 1,000,000 | 20,000 | 1,000,000 | 0.79 | 338 | 116 |
| KD-tree | 5,000,000 | 20,000 | 200,000 | 3.40 | 483 | 78 |
| KD-tree, Parquet in/out | 1,000,000 | 20,000 | 200,000 | 1.72 | 365 | 75 |

The cross join materialises places x terminals rows, about 110 bytes per pair here. At
1,000,000 x 20,000 that is 20 billion pairs, roughly 2 TB, and around 75 minutes of compute even
if it were batched. The KD-tree does the same 1M x 20k job in under a second.

`find` on the Staten Island bus feed (805,449 `stop_times` rows) takes about 2 s; peak RSS is
292 MB with `--chunk 100000` and 461 MB with `--chunk 1000000`.

## Limitations

- Detection is frequency based. A terminal with little service (a weekly ferry, a rural bus
  terminus) can fall below both thresholds, and a busy through-station where few trips start or
  end can be missed. Lower `--min-trips` / `--min-routes` for sparse networks.
- Counts are trips in the feed, not trips per day. Feeds that list many service calendars will
  show larger numbers than feeds that list one week; compare counts within a feed, not across feeds.
- Stops without a `parent_station` are not clustered within a feed. Staten Island's St. George
  ferry terminal appears as separate bus ramps (A, C, D) because the feed models them that way.
- Cross-feed dedup needs matching names. "Central Station" and "Central Stn" within 50 m stay
  separate.
- `reverse_geocoder` assigns the admin region of the nearest GeoNames town, which can be wrong near
  borders: St. George on Staten Island comes back as New Jersey.
- Random uniform points are close to a best case for the KD-tree. Real places cluster, which costs
  somewhat more per query; the memory profile does not change.

## Tests

```
pytest
ruff check .
```

The tests build small GTFS zips in a temp directory covering parent/child stops, a loop route, a
feed without `parent_station`, a ferry route, a BOM-prefixed nested feed, cross-feed dedup, and
compare nearest-terminal results against brute-force haversine on random points for several chunk
sizes.

## License

MIT
