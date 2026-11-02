"""Prompts for the NL-to-Cypher assistant."""

import json

from app.database.schema import schema_as_text

CYPHER_SYSTEM = f"""You translate questions about a movie-ratings graph into ONE read-only Neo4j Cypher query.

{schema_as_text()}

Facts about the data:
- Movie.genres is a list of strings, e.g. ['Action', 'Sci-Fi']. Genre names: Action, Adventure, Animation, Children,
  Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, IMAX, Musical, Mystery, Romance, Sci-Fi, Thriller,
  War, Western.
- Movie.title has no year, e.g. 'The Matrix'. Match a title with toLower(m.title) = '...' when the question
  names an exact movie, and with CONTAINS for partial names.
- Movie.year is an integer. A decade like 'the 90s' means m.year >= 1990 AND m.year <= 1999.
- RATED.rating is 0.5 to 5.0. A rating >= 4.0 means the user liked the movie.
- Movie.avgRating and Movie.ratingCount are precomputed. Prefer movies with ratingCount >= 20 for "best" questions.
- SIMILAR links movies liked by the same people; higher score = more similar.
- Tag.name is lower-case, e.g. 'pixar', 'dark comedy'.

Rules:
- Only MATCH / OPTIONAL MATCH / WHERE / WITH / UNWIND / RETURN / ORDER BY / SKIP / LIMIT. Never modify data.
- Use only the labels, relationship types and properties listed above.
- Always end with LIMIT (at most 50).
- Return readable columns with aliases (e.g. RETURN m.title AS title, m.year AS year).
- The graph has NO data about box office, budget, revenue, directors, actors, cast, runtime, awards, plot or
  country. If answering needs such data, do NOT substitute another property: reply with "cypher": "" and say
  in "reasoning" which data is missing.

Reply with JSON: {{"cypher": "...", "reasoning": "one short sentence"}}"""

FEW_SHOT = [
    (
        "What are the 5 highest rated comedies with at least 50 ratings?",
        "MATCH (m:Movie) WHERE 'Comedy' IN m.genres AND m.ratingCount >= 50 "
        "RETURN m.title AS title, m.year AS year, m.avgRating AS avgRating ORDER BY m.avgRating DESC LIMIT 5",
    ),
    (
        "Best horror movies of the 80s?",
        "MATCH (m:Movie) WHERE 'Horror' IN m.genres AND m.year >= 1980 AND m.year <= 1989 AND m.ratingCount >= 20 "
        "RETURN m.title AS title, m.year AS year, m.avgRating AS avgRating ORDER BY m.avgRating DESC LIMIT 10",
    ),
    (
        "Which movies are most similar to Toy Story?",
        "MATCH (m:Movie)-[s:SIMILAR]->(o:Movie) WHERE toLower(m.title) = 'toy story' "
        "RETURN o.title AS title, round(s.score, 3) AS similarity ORDER BY s.score DESC LIMIT 10",
    ),
    (
        "Which actors appear in Titanic?",
        "",  # the graph has no cast data: the model must decline instead of guessing
    ),
    (
        "How many movies did user 42 rate 5 stars?",
        "MATCH (:User {userId: 42})-[r:RATED]->(m:Movie) WHERE r.rating = 5.0 "
        "RETURN count(m) AS fiveStarMovies LIMIT 1",
    ),
    (
        "Which genres does user 7 like most?",
        "MATCH (:User {userId: 7})-[r:RATED]->(:Movie)-[:IN_GENRE]->(g:Genre) WHERE r.rating >= 4.0 "
        "RETURN g.name AS genre, count(*) AS likedMovies ORDER BY likedMovies DESC LIMIT 5",
    ),
]

ANSWER_SYSTEM = """You answer a question using ONLY the database rows provided.
- Do not add facts, numbers, plot details or opinions that are not in the rows.
- If the rows do not answer the question, say so.
- Copy numbers exactly as they appear in the rows.
- Keep it to 1-4 sentences."""

CYPHER_SCHEMA = {
    "type": "object",
    "properties": {"cypher": {"type": "string"}, "reasoning": {"type": "string"}},
    "required": ["cypher", "reasoning"],
}


def cypher_messages(question: str, errors: list[str], previous: str | None) -> list[dict[str, str]]:
    """Build the chat for query generation, including feedback from a failed attempt."""
    messages = [{"role": "system", "content": CYPHER_SYSTEM}]
    for q, cypher in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        reasoning = "the graph has no data about actors" if not cypher else "example"
        messages.append({"role": "assistant", "content": json.dumps({"cypher": cypher, "reasoning": reasoning})})
    messages.append({"role": "user", "content": question})
    if previous and errors:
        messages.append({"role": "assistant", "content": json.dumps({"cypher": previous})})
        messages.append(
            {
                "role": "user",
                "content": "That query was rejected:\n- " + "\n- ".join(errors) + "\nWrite a corrected query.",
            }
        )
    return messages


def answer_messages(question: str, rows: list[dict]) -> list[dict[str, str]]:
    """Build the chat for summarising the verified rows."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Question: {question}\n\nRows (JSON):\n{rows}"},
    ]
