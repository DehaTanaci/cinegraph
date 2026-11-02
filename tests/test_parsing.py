"""Data-cleaning and feature helpers."""

import pytest

from app.database.schema import schema_as_text
from scripts.seed_data import batched, normalize_tag, parse_genres, parse_title


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("American President, The (1995)", ("The American President", 1995)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Stranger Things (2016-)", ("Stranger Things", 2016)),
        ("2001: A Space Odyssey (1968)", ("2001: A Space Odyssey", 1968)),
    ],
)
def test_parse_title(raw, expected):
    assert parse_title(raw) == expected


def test_parse_genres():
    assert parse_genres("Adventure|Comedy") == ["Adventure", "Comedy"]
    assert not parse_genres("(no genres listed)")


def test_normalize_tag():
    assert normalize_tag("  Dark   Comedy ") == "dark comedy"


def test_batched():
    assert [len(b) for b in batched(range(25), 10)] == [10, 10, 5]


def test_schema_text_mentions_everything():
    text = schema_as_text()
    for name in ("Movie", "User", "Genre", "Tag", "RATED", "IN_GENRE", "TAGGED_AS", "SIMILAR"):
        assert name in text
