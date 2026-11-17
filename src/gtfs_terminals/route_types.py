import pandas as pd

BASIC_ROUTE_TYPES = {
    0: "tram",
    1: "subway",
    2: "rail",
    3: "bus",
    4: "ferry",
    5: "cable_tram",
    6: "aerial_lift",
    7: "funicular",
    11: "trolleybus",
    12: "monorail",
}

# Many European feeds use the extended (HVT) codes, e.g. 700 for bus, 1200 for ferry.
# Grouping by the hundreds digit is coarse but is all the --mode filter needs.
EXTENDED_ROUTE_GROUPS = {
    1: "rail",
    2: "coach",
    3: "rail",
    4: "subway",
    5: "subway",
    6: "subway",
    7: "bus",
    8: "trolleybus",
    9: "tram",
    10: "ferry",
    11: "air",
    12: "ferry",
    13: "aerial_lift",
    14: "funicular",
    15: "taxi",
}


def route_type_label(code: int) -> str:
    if code in BASIC_ROUTE_TYPES:
        return BASIC_ROUTE_TYPES[code]
    return EXTENDED_ROUTE_GROUPS.get(code // 100, "other")


def label_route_types(codes: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(codes, errors="coerce")
    return numeric.map(lambda c: "other" if pd.isna(c) else route_type_label(int(c)))
