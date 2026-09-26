import re
import unicodedata
import pandas as pd


def normalize_text(text):
    """
    Normalize a text value.

    Steps:
    1. Handle missing values
    2. Unicode normalization
    3. Convert to lowercase
    4. Replace punctuation with spaces
    5. Normalize whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace punctuation/special characters with spaces
    # Keeps Unicode characters such as Hindi and accented letters
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def add_normalized_columns(df):
    """
    Add normalized columns without changing
    the original columns.
    """

    df = df.copy()

    # Normalized business name
    df["name_norm"] = df["business_name"].apply(normalize_text)

    # Normalized business address
    df["address_norm"] = df["business_address"].apply(normalize_text)

    # Normalized country
    df["country_norm"] = df["country"].apply(normalize_text)

    # Remove spaces from normalized name
    df["name_compact"] = df["name_norm"].str.replace(
        " ", "", regex=False
    )

    # Remove spaces from normalized address
    df["address_compact"] = df["address_norm"].str.replace(
        " ", "", regex=False
    )

    return df