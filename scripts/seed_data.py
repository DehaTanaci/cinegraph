"""Load a MovieLens dataset into Neo4j.

Usage:
    python -m scripts.seed_data                 # uses DATASET / DATA_DIR from .env
    python -m scripts.seed_data --dataset ml-25m --max-ratings 2000000

Cleaning steps (all documented in the README):
- the release year is parsed out of the title ("Toy Story (1995)" -> title "Toy Story", year 1995);
- trailing articles are restored ("Matrix, The" -> "The Matrix");
- "(no genres listed)" becomes an empty genre list, not a fake Genre node;
- tags are lower-cased and trimmed so "Funny" and "funny " are the same Tag;
- only tags used on a movie are kept, aggregated with a usage count.
"""

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient
from app.database.schema import apply_schema

BATCH_SIZE = 10_000
TITLE_YEAR = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})(?:[-–]\d{0,4})?\)\s*$")
TRAILING_ARTICLE = re.compile(r"^(?P<rest>.*), (?P<article>The|A|An|Les|Le|La|L'|Il|Das|Der|Die|El)$")
NO_GENRES = "(no genres listed)"


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split 'Matrix, The (1999)' into ('The Matrix', 1999)."""
    raw = raw.strip()
    match = TITLE_YEAR.match(raw)
    title, year = (match.group("title"), int(match.group("year"))) if match else (raw, None)
    article = TRAILING_ARTICLE.match(title)
    if article:
        sep = "" if article.group("article").endswith("'") else " "
        title = f"{article.group('article')}{sep}{article.group('rest')}"
    return title, year


def parse_genres(raw: str) -> list[str]:
    """'Adventure|Comedy' -> ['Adventure', 'Comedy']; '(no genres listed)' -> []."""
    if raw.strip() == NO_GENRES:
        return []
    return [g.strip() for g in raw.split("|") if g.strip()]


def normalize_tag(raw: str) -> str:
    """Lower-case and collapse whitespace so equivalent tags merge."""
    return re.sub(r"\s+", " ", raw.strip().lower())


def batched(rows: Iterable[dict], size: int = BATCH_SIZE) -> Iterator[list[dict]]:
    """Yield lists of at most `size` rows."""
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def read_csv(path: Path) -> Iterator[dict]:
    """Stream a CSV file as dicts."""
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load_movies(client: Neo4jClient, folder: Path) -> int:
    """Create Movie and Genre nodes and IN_GENRE relationships."""
    links = {row["movieId"]: row for row in read_csv(folder / "links.csv")}
    rows = []
    for row in read_csv(folder / "movies.csv"):
        title, year = parse_title(row["title"])
        link = links.get(row["movieId"], {})
        rows.append(
            {
                "movieId": int(row["movieId"]),
                "title": title,
                "year": year,
                "genres": parse_genres(row["genres"]),
                "imdbId": link.get("imdbId") or None,
                "tmdbId": int(link["tmdbId"]) if link.get("tmdbId") else None,
            }
        )
    for batch in batched(rows):
        client.write(
            """
            UNWIND $rows AS row
            MERGE (m:Movie {movieId: row.movieId})
            SET m.title = row.title, m.year = row.year, m.genres = row.genres,
                m.imdbId = row.imdbId, m.tmdbId = row.tmdbId
            WITH m, row
            UNWIND row.genres AS genreName
            MERGE (g:Genre {name: genreName})
            MERGE (m)-[:IN_GENRE]->(g)
            """,
            {"rows": batch},
        )
    return len(rows)


def load_ratings(client: Neo4jClient, folder: Path, max_ratings: int | None) -> int:
    """Create User nodes and RATED relationships, in batches."""
    total = 0

    def rows():
        for i, row in enumerate(read_csv(folder / "ratings.csv")):
            if max_ratings is not None and i >= max_ratings:
                return
            yield {
                "userId": int(row["userId"]),
                "movieId": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            }

    for batch in batched(rows()):
        client.write(
            """
            UNWIND $rows AS row
            MATCH (m:Movie {movieId: row.movieId})
            MERGE (u:User {userId: row.userId})
            MERGE (u)-[r:RATED]->(m)
            SET r.rating = row.rating, r.timestamp = datetime({epochSeconds: row.timestamp})
            """,
            {"rows": batch},
        )
        total += len(batch)
        print(f"  ratings loaded: {total:,}", end="\r", flush=True)
    print()
    return total


def load_tags(client: Neo4jClient, folder: Path) -> int:
    """Aggregate user tags into (:Movie)-[:TAGGED_AS {count}]->(:Tag)."""
    path = folder / "tags.csv"
    if not path.exists():
        return 0
    counts: Counter = Counter()
    for row in read_csv(path):
        tag = normalize_tag(row["tag"])
        if tag:
            counts[(int(row["movieId"]), tag)] += 1
    rows = [{"movieId": m, "tag": t, "count": c} for (m, t), c in counts.items()]
    for batch in batched(rows):
        client.write(
            """
            UNWIND $rows AS row
            MATCH (m:Movie {movieId: row.movieId})
            MERGE (t:Tag {name: row.tag})
            MERGE (m)-[r:TAGGED_AS]->(t)
            SET r.count = row.count
            """,
            {"rows": batch},
        )
    return len(rows)


def compute_aggregates(client: Neo4jClient) -> None:
    """Pre-compute rating stats so list endpoints never aggregate millions of edges per request."""
    client.write(
        """
        MATCH (m:Movie)
        CALL (m) {
            OPTIONAL MATCH (m)<-[r:RATED]-()
            RETURN count(r) AS cnt, avg(r.rating) AS avgRating
        }
        SET m.ratingCount = cnt, m.avgRating = round(coalesce(avgRating, 0.0), 2)
        """
    )
    client.write(
        """
        MATCH (u:User)
        SET u.ratingCount = COUNT { (u)-[:RATED]->() }
        """
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=settings.dataset)
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--max-ratings", type=int, default=None, help="cap ratings (useful for ml-25m)")
    parser.add_argument("--reset", action="store_true", help="delete all existing data first")
    args = parser.parse_args(argv)

    folder = Path(args.data_dir) / args.dataset
    if not (folder / "movies.csv").exists():
        sys.exit(f"Dataset not found in {folder}. Run `make data` first.")

    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        start = time.perf_counter()
        if args.reset:
            print("Deleting existing data...")
            client.run_auto("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        apply_schema(client)
        print(f"Movies: {load_movies(client, folder):,}")
        print(f"Ratings: {load_ratings(client, folder, args.max_ratings):,}")
        print(f"Movie-tag links: {load_tags(client, folder):,}")
        compute_aggregates(client)
        print(f"Done in {time.perf_counter() - start:.1f}s")
    finally:
        client.close()


if __name__ == "__main__":
    main()
