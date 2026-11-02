"""Train and evaluate the "will this user like this movie?" link-prediction model.

Evaluation is temporal, which is how a recommender is used in reality: we learn from the past
and predict the future.

1. For every user, hide their most recent 20% of likes (leave-last-out). The rest = training graph.
   (A single global time cut does not work on MovieLens: most late likes belong to users who did
   not exist yet before the cut, so there is nothing to learn about them.)
2. Compute FastRP embeddings on the training graph only (so the test edges are invisible to it).
3. Positive examples = likes; negative examples = random movies the user never rated (1:1).
4. Train logistic regression on pair features; test on likes that happened after the cut.
5. Compare against a popularity baseline (score = how many people liked the movie before the cut).

Then a final model is retrained on embeddings of the full graph (written by `make gds`) and saved
for the /predict endpoint. Metrics are written to docs/ml_report.md.
"""

import json
import random
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient
from app.services.gds_service import EMBEDDING_DIM, RANDOM_SEED
from app.services.prediction_service import MODEL_FILE, pair_features

# scikit-learn passes an option newer SciPy no longer knows; the warning is harmless noise.
warnings.filterwarnings("ignore", message="Unknown solver options")

TRAIN_GRAPH = "likes_train"
REPORT = Path("docs/ml_report.md")


def liked_edges(client: Neo4jClient, like: float) -> list[dict]:
    """All liked ratings with epoch-second timestamps."""
    return client.read(
        """
        MATCH (u:User)-[r:RATED]->(m:Movie) WHERE r.rating >= $like
        RETURN u.userId AS u, m.movieId AS m, r.timestamp.epochSeconds AS t
        """,
        {"like": like},
    )


def split_per_user(edges: list[dict], test_share: float = 0.2) -> tuple[list, list]:
    """Per user, the most recent `test_share` of likes go to test (users with < 5 likes stay in train)."""
    by_user: dict[int, list[dict]] = {}
    for edge in edges:
        by_user.setdefault(edge["u"], []).append(edge)
    train, test = [], []
    for user_edges in by_user.values():
        user_edges.sort(key=lambda e: (e["t"], e["m"]))
        n_test = int(len(user_edges) * test_share) if len(user_edges) >= 5 else 0
        cut = len(user_edges) - n_test
        train += [(e["u"], e["m"]) for e in user_edges[:cut]]
        test += [(e["u"], e["m"]) for e in user_edges[cut:]]
    return train, test


def train_embeddings(client: Neo4jClient, train_edges: list[tuple[int, int]]) -> tuple[dict, dict]:
    """FastRP embeddings computed only from the training likes."""
    client.write("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", {"g": TRAIN_GRAPH})
    client.write(
        """
        UNWIND $edges AS e
        MATCH (u:User {userId: e[0]}), (m:Movie {movieId: e[1]})
        WITH gds.graph.project($g, u, m, {sourceNodeLabels: labels(u), targetNodeLabels: labels(m)},
                               {undirectedRelationshipTypes: ['*']}) AS g
        RETURN g.nodeCount AS n
        """,
        {"edges": [list(e) for e in train_edges], "g": TRAIN_GRAPH},
    )
    rows = client.read(
        """
        CALL gds.fastRP.stream($g, {embeddingDimension: $dim, randomSeed: $seed,
                                    iterationWeights: [0.0, 1.0, 1.0], concurrency: 1})
        YIELD nodeId, embedding
        WITH gds.util.asNode(nodeId) AS n, embedding
        RETURN n.userId AS u, n.movieId AS m, embedding
        """,
        {"g": TRAIN_GRAPH, "dim": EMBEDDING_DIM, "seed": RANDOM_SEED},
    )
    client.write("CALL gds.graph.drop($g, false) YIELD graphName RETURN graphName", {"g": TRAIN_GRAPH})
    users = {r["u"]: np.array(r["embedding"]) for r in rows if r["u"] is not None}
    movies = {r["m"]: np.array(r["embedding"]) for r in rows if r["m"] is not None}
    return users, movies


def with_negatives(positives: list[tuple[int, int]], movie_pool: list[int], rated: set, rng: random.Random):
    """Add one random never-rated movie per positive pair."""
    pairs, labels = [], []
    for user, movie in positives:
        pairs.append((user, movie))
        labels.append(1)
        while True:
            candidate = rng.choice(movie_pool)
            if (user, candidate) not in rated:
                pairs.append((user, candidate))
                labels.append(0)
                break
    return pairs, np.array(labels)


def featurise(pairs, users, movies, likes: dict[int, int]) -> np.ndarray:
    """Feature matrix for (user, movie) pairs."""
    return np.vstack([pair_features(users[u], movies[m], likes.get(m, 0)) for u, m in pairs])


def like_counts(edges: list[tuple[int, int]]) -> dict[int, int]:
    """How many likes each movie has among `edges`."""
    counts: dict[int, int] = {}
    for _, movie in edges:
        counts[movie] = counts.get(movie, 0) + 1
    return counts


def evaluate(client: Neo4jClient, like: float) -> dict:
    """Temporal hold-out evaluation. Returns the metrics dict."""
    rng = random.Random(RANDOM_SEED)
    edges = liked_edges(client, like)
    rated = {(r["u"], r["m"]) for r in client.read("MATCH (u:User)-[:RATED]->(m:Movie) RETURN u.userId AS u, "
                                                   "m.movieId AS m")}
    train_edges, held_out = split_per_user(edges)
    users, movies = train_embeddings(client, train_edges)
    test_edges = [(u, m) for u, m in held_out if u in users and m in movies]
    pool = sorted(movies)
    popularity = like_counts(train_edges)

    x_train_pairs, y_train = with_negatives(train_edges, pool, rated, rng)
    x_test_pairs, y_test = with_negatives(test_edges, pool, rated, rng)
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(featurise(x_train_pairs, users, movies, popularity), y_train)
    scores = model.predict_proba(featurise(x_test_pairs, users, movies, popularity))[:, 1]
    baseline = np.array([popularity.get(m, 0) for _, m in x_test_pairs], dtype=float)

    return {
        "split": "per-user leave-last-20%-out",
        "train_positive_edges": len(train_edges),
        "test_positive_edges": len(test_edges),
        "test_dropped_cold_start": len(held_out) - len(test_edges),
        "model_auc": round(roc_auc_score(y_test, scores), 4),
        "model_average_precision": round(average_precision_score(y_test, scores), 4),
        "popularity_auc": round(roc_auc_score(y_test, baseline), 4),
        "popularity_average_precision": round(average_precision_score(y_test, baseline), 4),
    }


def train_final(client: Neo4jClient, like: float, model_dir: str) -> int:
    """Retrain on the full graph's stored embeddings and save the model for the API."""
    rng = random.Random(RANDOM_SEED)
    rows = client.read(
        """
        MATCH (n) WHERE n.embedding IS NOT NULL
        RETURN n.userId AS u, n.movieId AS m, n.embedding AS e
        """
    )
    users = {r["u"]: np.array(r["e"]) for r in rows if r["u"] is not None}
    movies = {r["m"]: np.array(r["e"]) for r in rows if r["m"] is not None}
    if not users:
        raise SystemExit("No embeddings found. Run `make gds` first.")
    positives = [(e["u"], e["m"]) for e in liked_edges(client, like) if e["u"] in users and e["m"] in movies]
    rated = {(r["u"], r["m"]) for r in client.read("MATCH (u:User)-[:RATED]->(m:Movie) RETURN u.userId AS u, "
                                                   "m.movieId AS m")}
    pairs, labels = with_negatives(positives, sorted(movies), rated, rng)
    model = LogisticRegression(max_iter=2000, C=1.0).fit(featurise(pairs, users, movies, like_counts(positives)),
                                                         labels)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": "logistic_regression_fastrp64", "like_threshold": like},
                Path(model_dir) / MODEL_FILE)
    return len(pairs)


def write_report(metrics: dict) -> None:
    """Persist metrics as Markdown for the README."""
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(
        "# Link-prediction evaluation\n\n"
        "Task: predict whether a user will like (rate >= 4) a movie, using FastRP graph embeddings.\n"
        "Split: per user, their most recent 20% of likes are hidden from the graph and used as the test set.\n"
        "Negatives: one random movie the user never rated per positive.\n\n"
        "| Metric | FastRP + logistic regression | Popularity baseline |\n|---|---:|---:|\n"
        f"| ROC AUC | {metrics['model_auc']} | {metrics['popularity_auc']} |\n"
        f"| Average precision | {metrics['model_average_precision']} | {metrics['popularity_average_precision']} |\n\n"
        f"Training likes: {metrics['train_positive_edges']:,}. Test likes: {metrics['test_positive_edges']:,} "
        f"({metrics['test_dropped_cold_start']:,} test likes dropped because the movie had no like in the "
        "training graph — the cold-start problem, which graph embeddings cannot solve).\n\n"
        "Caveat: random negatives are mostly obscure movies, which flatters popularity; the baseline is "
        "therefore strong, and beating it is the meaningful test.\n\n"
        f"```json\n{json.dumps(metrics, indent=2)}\n```\n",
        encoding="utf-8",
    )


def main() -> None:
    """Evaluate, report, then train and save the serving model."""
    settings = get_settings()
    client = Neo4jClient(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    try:
        start = time.perf_counter()
        metrics = evaluate(client, settings.like_threshold)
        print(json.dumps(metrics, indent=2))
        write_report(metrics)
        n = train_final(client, settings.like_threshold, settings.model_dir)
        print(f"Final model trained on {n:,} pairs, saved to {settings.model_dir}/{MODEL_FILE} "
              f"({time.perf_counter() - start:.1f}s)")
    finally:
        client.close()


if __name__ == "__main__":
    main()
