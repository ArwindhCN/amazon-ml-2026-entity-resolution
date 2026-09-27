"""
Text cleaning for business names and addresses (v1).

Works on whole pandas columns at once (vectorized), so millions of rows take seconds.
Nothing here is specific to one country: it works on any language and script.

Run this file directly to see before/after examples:
    .venv/bin/python code/business_entity_resolution/src/normalize.py
"""
import pandas as pd

# NFKD splits an accented letter into letter + separate accent mark (é -> e + ´).
# This range is those separate accent marks; deleting them strips accents.
# Indian-script vowel signs live in other Unicode ranges, so they are untouched.
ACCENTS = r"[̀-ͯ]"

# Anything that is not a letter (\pL), digit (\pN) or script mark (\pM), in any language.
NOT_WORD = r"[^\pL\pN\pM]+"

# Placeholder words that mean "no value" in addresses.
PLACEHOLDERS = r"\b(null|n/a|none)\b"


def _base(s):
    """Unicode-normalize, strip accents, lowercase."""
    return (s.fillna("")
             .str.normalize("NFKD")
             .str.replace(ACCENTS, "", regex=True)
             .str.lower())


def _squash(s):
    """Turn every run of punctuation/space into one space, trim the ends."""
    return s.str.replace(NOT_WORD, " ", regex=True).str.strip()


# Digits used as look-alike letters inside words: "n0rthern", "5ecurity", "wi1lis", "6olden".
LOOKALIKE = str.maketrans("0134568", "oleasgb")


def _fix_word(w):
    # Only words that mix letters and digits and are 4+ long: leaves "3204", "1st", "3m" alone.
    if len(w) >= 4 and any(c.isdigit() for c in w) and any(c.isalpha() for c in w):
        return w.translate(LOOKALIKE)
    return w


def fix_lookalikes(s):
    mixed = s.str.contains(r"[a-z][0-9]|[0-9][a-z]", regex=True)  # only these rows need Python
    s = s.copy()
    s[mixed] = s[mixed].map(lambda t: " ".join(_fix_word(w) for w in t.split()))
    return s


def clean_name(s):
    return fix_lookalikes(_squash(_base(s).str.replace("&", " and ", regex=False)))


def clean_address(s):
    return _squash(_base(s).str.replace(PLACEHOLDERS, " ", regex=True))


INDIC = r"[ऀ-෿]"


def transliterate(names_c, mapping):
    """Replace Indian-script words by their learned Latin spelling (learn_translit.py).

    "ಸಿಲ್ವರ್ ಪ್ರೊಡಕ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" -> "silver products private limited".
    Only rows containing Indian script are touched; unknown words are kept as they are.
    """
    has = names_c.str.contains(INDIC, regex=True)
    out = names_c.copy()
    out[has] = out[has].map(lambda t: " ".join(mapping.get(w, w) for w in t.split()))
    return out


def load_clean(cache, name, country=None, translit=None):
    """Load cache/<name>.parquet (optionally one country only) with cleaned name and address.

    translit: optional {indian_word: latin_word} dictionary applied to the cleaned names.
    """
    filters = [("country", "==", country)] if country is not None else None
    df = pd.read_parquet(f"{cache}/{name}.parquet", filters=filters)
    name_c = clean_name(df["business_name"])
    indic_src = name_c.str.contains(INDIC, regex=True).to_numpy()   # before any translation
    if translit:
        name_c = transliterate(name_c, translit)
    return pd.DataFrame({"entity_id": df["entity_id"], "country": df["country"],
                         "name_c": name_c, "addr_c": clean_address(df["business_address"]),
                         "indic_src": indic_src})


if __name__ == "__main__":
    names = pd.Series([
        "Riki F. Rivera, O.D. Inc.", "-- RIKI F. Rivera, O.D.", "Esquivel + Vffnk Illumination P.C.",
        "Marshall & Johnson P.C.", "... Great Trúst Group", "ALL-SEMICONDUCTOR CONSULTANTS,-LLC",
        "एपेक्स पावर लिमिटेड", "Tournesol & Cie Sàrl", "Rkk Products Private Limited",
        "N0rthern One", "Leonard 5ecurity LLC", "Wi1lis (Center)", "#6 Brewing Inc", "3M Company",
    ], dtype=str)
    addrs = pd.Series([
        "97 LIEBRTY ST, DANBURY CITY, CT", "##7601 Beach Dr, OAK ISLAND, NC",
        "#7 KAYLAN DR, NULL, MORGANTOWN, WV", "6S 155 W RIDGE DRIVE, N/A, AURORA, IL",
        "003810 Beresford Road, Charlotte, North Carolina", "No.68/16 First Floor, 4Th Street, Chennai",
        "791, CHANAKYAPURI SECTOR 3, RLY CRSG, AHMEDAABD, ગુજરાત", "Nº 257 Route De La Côte D'amour, Saint-nazaire",
    ], dtype=str)
    print("NAMES")
    for a, b in zip(names, clean_name(names)):
        print(f"  {a:40s} -> {b}")
    print("ADDRESSES")
    for a, b in zip(addrs, clean_address(addrs)):
        print(f"  {a:52s} -> {b}")
