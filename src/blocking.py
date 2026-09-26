import pandas as pd
import re


def add_blocking_keys(df):
    """
    Create blocking keys used to retrieve candidate matches.

    The original columns are not modified.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # 1. Exact normalized name
    # ---------------------------------------------------------
    df["block_name"] = (
        df["country_norm"].fillna("")
        + "|"
        + df["name_norm"].fillna("")
    )

    # ---------------------------------------------------------
    # 2. Compact normalized name
    # ---------------------------------------------------------
    df["block_name_compact"] = (
        df["country_norm"].fillna("")
        + "|"
        + df["name_compact"].fillna("")
    )

    # ---------------------------------------------------------
    # 3. Name prefix
    # ---------------------------------------------------------
    df["name_prefix"] = df["name_compact"].str[:8]

    df["block_name_prefix"] = (
        df["country_norm"].fillna("")
        + "|"
        + df["name_prefix"].fillna("")
    )

    # ---------------------------------------------------------
    # 4. Extract house number from address
    # ---------------------------------------------------------
    df["house_number"] = (
        df["address_norm"]
        .str.extract(r"^\s*(\d+)", expand=False)
        .fillna("")
    )

    df["block_house_number"] = (
        df["country_norm"].fillna("")
        + "|"
        + df["house_number"]
    )

    return df


def build_indexes(source_df):
    """
    Build lookup dictionaries for fast candidate retrieval.

    Returns:
        Dictionary containing multiple blocking indexes.
    """

    indexes = {}

    # ---------------------------------------------------------
    # Exact normalized name
    # ---------------------------------------------------------
    indexes["name"] = (
        source_df[
            source_df["name_norm"].ne("")
        ]
        .groupby("block_name")["entity_id"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Compact name
    # ---------------------------------------------------------
    indexes["name_compact"] = (
        source_df[
            source_df["name_compact"].ne("")
        ]
        .groupby("block_name_compact")["entity_id"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # Name prefix
    # ---------------------------------------------------------
    indexes["name_prefix"] = (
        source_df[
            source_df["name_prefix"].str.len() >= 5
        ]
        .groupby("block_name_prefix")["entity_id"]
        .apply(list)
        .to_dict()
    )

    # ---------------------------------------------------------
    # House number
    # ---------------------------------------------------------
    indexes["house_number"] = (
        source_df[
            source_df["house_number"].ne("")
        ]
        .groupby("block_house_number")["entity_id"]
        .apply(list)
        .to_dict()
    )

    return indexes


def get_candidates(row, indexes):
    """
    Retrieve candidate entity IDs for one Source 1 record.

    Candidates from different blocking strategies
    are UNIONED together.
    """

    candidates = set()

    country = row["country_norm"]

    # ---------------------------------------------------------
    # Block 1: Exact normalized name
    # ---------------------------------------------------------
    name = row["name_norm"]

    if name:
        key = f"{country}|{name}"
        candidates.update(
            indexes["name"].get(key, [])
        )

    # ---------------------------------------------------------
    # Block 2: Compact name
    # ---------------------------------------------------------
    name_compact = row["name_compact"]

    if name_compact:
        key = f"{country}|{name_compact}"
        candidates.update(
            indexes["name_compact"].get(key, [])
        )

    # ---------------------------------------------------------
    # Block 3: Name prefix
    # ---------------------------------------------------------
    prefix = row["name_prefix"]

    if len(prefix) >= 5:
        key = f"{country}|{prefix}"
        candidates.update(
            indexes["name_prefix"].get(key, [])
        )

    # ---------------------------------------------------------
    # Block 4: House number
    # ---------------------------------------------------------
    house_number = row["house_number"]

    if house_number:
        key = f"{country}|{house_number}"
        candidates.update(
            indexes["house_number"].get(key, [])
        )

    return candidates