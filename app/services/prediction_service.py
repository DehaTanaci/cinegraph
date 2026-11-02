"""Serve the link-prediction model: probability that a user will like a movie."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient
from app.services.movie_service import NotFoundError, PrerequisiteError

MODEL_FILE = "link_model.joblib"


def pair_features(user_vec: np.ndarray, movie_vec: np.ndarray, movie_likes: int) -> np.ndarray:
    """Features for one (user, movie) pair.

    - element-wise (Hadamard) product of the two embeddings: the standard way to turn two node
      embeddings into an edge representation (large where both are strong on the same dimension);
    - cosine similarity of the embeddings (overall taste match);
    - log of how many people liked the movie (popularity — a strong signal on its own).
    """
    user_vec, movie_vec = np.asarray(user_vec, dtype=float), np.asarray(movie_vec, dtype=float)
    norm = np.linalg.norm(user_vec) * np.linalg.norm(movie_vec)
    cosine = float(user_vec @ movie_vec / norm) if norm else 0.0
    return np.concatenate([user_vec * movie_vec, [cosine, np.log1p(movie_likes)]])


class PredictionService:
    """Loads the trained model lazily and scores pairs using embeddings stored on the nodes."""

    _cache: dict[str, Any] = {}

    def __init__(self, client: Neo4jClient):
        self.client = client
        self.path = Path(get_settings().model_dir) / MODEL_FILE

    def _model(self) -> dict[str, Any]:
        key = str(self.path)
        if key not in self._cache:
            if not self.path.exists():
                raise PrerequisiteError("No trained model found. Run `make gds` then `make train`.")
            self._cache[key] = joblib.load(self.path)
        return self._cache[key]

    def predict(self, user_id: int, movie_id: int) -> dict[str, Any]:
        """Probability that `user_id` would rate `movie_id` >= the like threshold."""
        bundle = self._model()
        rows = self.client.read(
            """
            OPTIONAL MATCH (u:User {userId: $uid})
            OPTIONAL MATCH (m:Movie {movieId: $mid})
            RETURN u IS NOT NULL AS userExists, m IS NOT NULL AS movieExists,
                   u.embedding AS ue, m.embedding AS me,
                   COUNT { (m)<-[r:RATED]-() WHERE r.rating >= $like } AS likes
            """,
            {"uid": user_id, "mid": movie_id, "like": bundle.get("like_threshold", get_settings().like_threshold)},
        )[0]
        if not rows["userExists"]:
            raise NotFoundError(f"User {user_id} not found")
        if not rows["movieExists"]:
            raise NotFoundError(f"Movie {movie_id} not found")
        if rows["ue"] is None or rows["me"] is None:
            raise PrerequisiteError("This user or movie has no embedding (no liked ratings in the graph).")
        features = pair_features(rows["ue"], rows["me"], rows["likes"]).reshape(1, -1)
        probability = float(bundle["model"].predict_proba(features)[0, 1])
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "probability": round(probability, 4),
            "model": bundle.get("name", "logistic_regression_fastrp"),
        }
