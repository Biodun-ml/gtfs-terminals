import pandas as pd


class GeocoderUnavailable(RuntimeError):
    pass


def add_admin_regions(terminals: pd.DataFrame) -> pd.DataFrame:
    try:
        import reverse_geocoder
    except ImportError as exc:
        raise GeocoderUnavailable(
            "reverse_geocoder is not installed; install with `pip install gtfs-terminals[geo]`"
        ) from exc

    if terminals.empty:
        return terminals.assign(country_code=pd.Series(dtype=str), admin1=pd.Series(dtype=str))
    coords = list(zip(terminals["lat"], terminals["lon"], strict=True))
    # mode=1 stays single-process; the default spins up a process pool, which costs more than
    # it saves for the few thousand points a terminal list usually has.
    results = reverse_geocoder.search(coords, mode=1, verbose=False)
    return terminals.assign(
        country_code=[r["cc"] for r in results],
        admin1=[r["admin1"] for r in results],
    )
