import numpy as np

EARTH_RADIUS_KM = 6371.0088


def unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def km_to_chord(km: float | np.ndarray) -> float | np.ndarray:
    return 2.0 * np.sin(np.asarray(km) / (2.0 * EARTH_RADIUS_KM))


def chord_to_km(chord: float | np.ndarray) -> float | np.ndarray:
    half = np.clip(np.asarray(chord) / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(half)


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
