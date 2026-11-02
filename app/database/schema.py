"""Graph schema: constraints, indexes, and the description given to the LLM.

Constraints double as indexes: a uniqueness constraint on :Movie(movieId) means every
"find movie 42" lookup is an index seek instead of a scan over all movies.
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
    "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "CREATE CONSTRAINT genre_name IF NOT EXISTS FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE",
]

INDEXES = [
    # Title search ("toy story") — full-text so partial words and any casing match.
    "CREATE FULLTEXT INDEX movie_title_ft IF NOT EXISTS FOR (m:Movie) ON EACH [m.title]",
    # Range filters used by trending / "movies from the 90s" questions.
    "CREATE INDEX movie_year IF NOT EXISTS FOR (m:Movie) ON (m.year)",
    "CREATE INDEX movie_rating_count IF NOT EXISTS FOR (m:Movie) ON (m.ratingCount)",
    # "Discover" groups movies by taste community computed by Louvain.
    "CREATE INDEX movie_community IF NOT EXISTS FOR (m:Movie) ON (m.community)",
    # Relationship property index: recent ratings for trending.
    "CREATE INDEX rated_timestamp IF NOT EXISTS FOR ()-[r:RATED]-() ON (r.timestamp)",
]

# Everything the LLM is allowed to know about (and reference in) a query.
LLM_SCHEMA = {
    "nodes": {
        "Movie": ["movieId", "title", "year", "genres", "avgRating", "ratingCount", "pagerank", "community"],
        "User": ["userId", "ratingCount", "community"],
        "Genre": ["name"],
        "Tag": ["name"],
    },
    "relationships": {
        "RATED": {"from": "User", "to": "Movie", "properties": ["rating", "timestamp"]},
        "IN_GENRE": {"from": "Movie", "to": "Genre", "properties": []},
        "TAGGED_AS": {"from": "Movie", "to": "Tag", "properties": ["count"]},
        "SIMILAR": {"from": "Movie", "to": "Movie", "properties": ["score"]},
    },
}


def schema_as_text() -> str:
    """Render LLM_SCHEMA as compact text for prompts."""
    lines = ["Node labels and properties:"]
    for label, props in LLM_SCHEMA["nodes"].items():
        lines.append(f"  (:{label} {{{', '.join(props)}}})")
    lines.append("Relationships:")
    for rel, spec in LLM_SCHEMA["relationships"].items():
        props = f" {{{', '.join(spec['properties'])}}}" if spec["properties"] else ""
        lines.append(f"  (:{spec['from']})-[:{rel}{props}]->(:{spec['to']})")
    return "\n".join(lines)


def apply_schema(client) -> None:
    """Create all constraints and indexes (idempotent)."""
    for statement in CONSTRAINTS + INDEXES:
        client.write(statement)
