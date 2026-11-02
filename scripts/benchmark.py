"""Before/after benchmarks for indexes and query rewrites, using PROFILE.

For each case we measure database hits (work done by the engine — stable across machines)
and median wall-clock latency over several runs, then record the plan operators so the
README can show e.g. NodeByLabelScan -> NodeIndexSeekByRange.

Run: python -m scripts.benchmark   (writes docs/benchmarks.md)
"""

import statistics
import time
from pathlib import Path

from app.config import get_settings
from app.database.neo4j_client import get_client

RUNS = 7
REPORT = Path("docs/benchmarks.md")

INDEX_CASES = [
    {
        "name": "Movies of a decade (range filter on year)",
        "drop": "DROP INDEX movie_year IF EXISTS",
        "create": "CREATE INDEX movie_year IF NOT EXISTS FOR (m:Movie) ON (m.year)",
        "query": "MATCH (m:Movie) WHERE m.year >= 1990 AND m.year <= 1999 RETURN m.title, m.year",
    },
    {
        "name": "Ratings in the last 90 days (relationship index on RATED.timestamp)",
        "drop": "DROP INDEX rated_timestamp IF EXISTS",
        "create": "CREATE INDEX rated_timestamp IF NOT EXISTS FOR ()-[r:RATED]-() ON (r.timestamp)",
        "query": "MATCH ()-[r:RATED]->(m:Movie) WHERE r.timestamp >= datetime('2018-06-24') "
                 "RETURN m.title, r.rating",
    },
    {
        "name": "User lookup by id (uniqueness constraint = index)",
        "drop": "DROP CONSTRAINT user_id IF EXISTS",
        "create": "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.userId IS UNIQUE",
        "query": "MATCH (u:User {userId: 414})-[r:RATED]->(m:Movie) RETURN m.title, r.rating",
    },
]

REWRITE_CASES = [
    {
        "name": "Recommendations: all peers vs. top-30 peers",
        "before": """
            MATCH (u:User {userId: 414})-[r1:RATED]->(:Movie)<-[r2:RATED]-(peer:User)
            WHERE r1.rating >= 4 AND r2.rating >= 4
            WITH u, peer, count(*) AS overlap
            MATCH (peer)-[r:RATED]->(rec:Movie) WHERE r.rating >= 4 AND NOT EXISTS { (u)-[:RATED]->(rec) }
            RETURN rec.title, sum(overlap) AS score ORDER BY score DESC LIMIT 10""",
        "after": """
            MATCH (u:User {userId: 414})-[r1:RATED]->(:Movie)<-[r2:RATED]-(peer:User)
            WHERE r1.rating >= 4 AND r2.rating >= 4
            WITH u, peer, count(*) AS overlap ORDER BY overlap DESC LIMIT 30
            MATCH (peer)-[r:RATED]->(rec:Movie) WHERE r.rating >= 4 AND NOT EXISTS { (u)-[:RATED]->(rec) }
            RETURN rec.title, sum(overlap) AS score ORDER BY score DESC LIMIT 10""",
    },
    {
        "name": "Movie list: aggregate ratings per request vs. precomputed avgRating",
        "before": """
            MATCH (m:Movie)<-[r:RATED]-()
            WITH m, avg(r.rating) AS avgRating, count(r) AS n WHERE n >= 50
            RETURN m.title, avgRating ORDER BY avgRating DESC LIMIT 10""",
        "after": """
            MATCH (m:Movie) WHERE m.ratingCount >= 50
            RETURN m.title, m.avgRating ORDER BY m.avgRating DESC LIMIT 10""",
    },
    {
        "name": "Two movies: Cartesian product vs. index-anchored pattern",
        "before": """
            MATCH (a:Movie), (b:Movie)
            WHERE a.title = 'Toy Story' AND b.title = 'The Godfather'
            MATCH p = shortestPath((a)-[:SIMILAR*..8]-(b)) RETURN length(p)""",
        "after": """
            MATCH (a:Movie {movieId: 1}), (b:Movie {movieId: 858})
            MATCH p = shortestPath((a)-[:SIMILAR*..8]-(b)) RETURN length(p)""",
    },
]


def measure(client, query: str) -> dict:
    """PROFILE once for db hits and operators, then time RUNS executions."""
    profile = client.profile(query)
    plan = client.explain(query)
    timings = []
    for _ in range(RUNS):
        start = time.perf_counter()
        client.read(query)
        timings.append((time.perf_counter() - start) * 1000)
    ops = [op.split("@")[0] for op in plan["operators"]]
    interesting = [op for op in ops if "Scan" in op or "Seek" in op or "Cartesian" in op]
    return {"db_hits": profile["db_hits"], "rows": profile["rows"], "median_ms": round(statistics.median(timings), 1),
            "operators": ", ".join(dict.fromkeys(interesting)) or ", ".join(dict.fromkeys(ops[:3]))}


def wait_for_indexes(client) -> None:
    """Block until all indexes are ONLINE."""
    client.run_auto("CALL db.awaitIndexes(300)")


def row(name: str, before: dict, after: dict) -> str:
    """One Markdown table row with the improvement factor."""
    factor = before["db_hits"] / after["db_hits"] if after["db_hits"] else float("inf")
    return (f"| {name} | {before['db_hits']:,} | {after['db_hits']:,} | **{factor:,.1f}×** | "
            f"{before['median_ms']} → {after['median_ms']} | `{before['operators']}` → `{after['operators']}` |")


def main() -> None:
    """Run all cases and write the report."""
    client = get_client()
    lines = [
        "# Query benchmarks (before / after)\n",
        f"Dataset: `{get_settings().dataset}`. Median of {RUNS} runs after one warm-up; db hits from `PROFILE`.\n",
        "## Indexes\n",
        "| Case | db hits before | db hits after | Less work | Latency ms | Plan |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for case in INDEX_CASES:
        client.write(case["drop"])
        client.read(case["query"])  # warm-up
        before = measure(client, case["query"])
        client.write(case["create"])
        wait_for_indexes(client)
        client.read(case["query"])
        after = measure(client, case["query"])
        lines.append(row(case["name"], before, after))
        print(case["name"], before, after)
    lines += ["\n## Query rewrites\n",
              "| Case | db hits before | db hits after | Less work | Latency ms | Plan |",
              "|---|---:|---:|---:|---:|---|"]
    for case in REWRITE_CASES:
        client.read(case["before"])
        client.read(case["after"])
        before, after = measure(client, case["before"]), measure(client, case["after"])
        lines.append(row(case["name"], before, after))
        print(case["name"], before, after)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT}")


if __name__ == "__main__":
    main()
