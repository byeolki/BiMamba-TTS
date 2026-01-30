import re

import inflect
from unidecode import unidecode

_inflect = inflect.engine()


_abbreviations = {
    "mr": "mister",
    "mrs": "misess",
    "dr": "doctor",
    "st": "saint",
    "co": "company",
    "jr": "junior",
    "maj": "major",
    "gen": "general",
    "drs": "doctors",
    "rev": "reverend",
    "lt": "lieutenant",
    "hon": "honorable",
    "sgt": "sergeant",
    "capt": "captain",
    "esq": "esquire",
    "ltd": "limited",
    "col": "colonel",
    "ft": "fort",
}


def expand_abbreviations(text):
    for abbr, full in _abbreviations.items():
        text = re.sub(f"\\b{abbr}\\.", full, text, flags=re.IGNORECASE)
    return text


def expand_numbers(text):
    def _expand_match(match):
        return _inflect.number_to_words(match.group(0))

    return re.sub(r"\d+", _expand_match, text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text)


def remove_punctuation(text):
    text = re.sub(r"[;:,.!?¿\-—]", "", text)
    return text


def english_cleaners(text):
    text = unidecode(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def get_text_cleaner(name):
    if name == "english_cleaners":
        return english_cleaners
    else:
        raise ValueError(f"Unknown text cleaner: {name}")
