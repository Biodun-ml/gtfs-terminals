import zipfile
from collections.abc import Callable, Iterator

import pandas as pd


class FeedError(ValueError):
    pass


def clean_header(name: str) -> str:
    # utf-8-sig only strips a BOM at the very start of the file; some exporters
    # put one in front of every header or pad names with spaces.
    return name.replace("﻿", "").strip()


def _member_name(zf: zipfile.ZipFile, filename: str) -> str | None:
    # Feeds are sometimes zipped with a top-level folder, so match on basename.
    for info in zf.infolist():
        if info.filename.rsplit("/", 1)[-1] == filename:
            return info.filename
    return None


def _usecols(columns: set[str] | None) -> Callable[[str], bool] | None:
    if columns is None:
        return None
    return lambda name: clean_header(name) in columns


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(columns=clean_header)
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    return frame


def has_table(zf: zipfile.ZipFile, filename: str) -> bool:
    return _member_name(zf, filename) is not None


def read_table(zf: zipfile.ZipFile, filename: str, columns: set[str] | None = None) -> pd.DataFrame:
    member = _member_name(zf, filename)
    if member is None:
        raise FeedError(f"{zf.filename}: no {filename}")
    with zf.open(member) as fh:
        frame = pd.read_csv(
            fh,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
            usecols=_usecols(columns),
        )
    return _tidy(frame)


def iter_table(
    zf: zipfile.ZipFile, filename: str, columns: set[str], chunksize: int
) -> Iterator[pd.DataFrame]:
    member = _member_name(zf, filename)
    if member is None:
        raise FeedError(f"{zf.filename}: no {filename}")
    with zf.open(member) as fh:
        reader = pd.read_csv(
            fh,
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
            usecols=_usecols(columns),
            chunksize=chunksize,
        )
        for chunk in reader:
            yield _tidy(chunk)


def require_columns(frame: pd.DataFrame, filename: str, required: set[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise FeedError(f"{filename} is missing columns: {', '.join(sorted(missing))}")
