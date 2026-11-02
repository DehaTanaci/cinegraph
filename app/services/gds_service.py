"""Graph Data Science jobs: projections and algorithms whose results are written back to the graph.

Pipeline (run with `make gds` or POST /api/admin/gds/run):
1. Node Similarity on movies, from shared "likers"  -> (:Movie)-[:SIMILAR {score}]->(:Movie)
2. Louvain on the undirected user-likes-movie graph -> `community` on User and Movie
3. PageRank on the SIMILAR network                  -> `pagerank` on Movie
4. FastRP embeddings on the likes graph             -> `embedding` on User and Movie (ML features)

Results are written as properties so API requests read precomputed values instead of
re-running algorithms on every call.
"""

import time
from typing import Any

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient

EMBEDDING_DIM = 64
RANDOM_SEED = 42


class GdsService:
    """Runs and persists the graph algorithms."""

    def __init__(self, client: Neo4jClient, like_threshold: float | None = None):
        self.client = client
        self.like = like_threshold if like_threshold is not None else get_settings().like_threshold

    def _drop(self, name: str) -> None:
        self.client.write("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", {"name": name})

    def project_likes(self, name: str, direction: str) -> dict[str, Any]:
        """Project only "liked" ratings into memory.

        direction="reverse": Movie -> User (movies are compared by who liked them).
        direction="undirected": both ways (for community detection and embeddings).
        """
        self._drop(name)
        if direction == "reverse":
            source, target, config = "m", "u", "{}"
        else:
            source, target, config = "u", "m", "{undirectedRelationshipTypes: ['*']}"
        rows = self.client.write(
            f"""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            WHERE r.rating >= $like
            WITH gds.graph.project($name, {source}, {target},
                 {{sourceNodeLabels: labels({source}), targetNodeLabels: labels({target})}},
                 {config}) AS g
            RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
            """,
            {"like": self.like, "name": name},
        )
        return rows[0]

    def node_similarity(self) -> dict[str, Any]:
        """Write the 10 most similar movies for each movie (Jaccard over shared likers)."""
        self.client.run_auto("MATCH ()-[s:SIMILAR]->() CALL (s) { DELETE s } IN TRANSACTIONS OF 10000 ROWS")
        self.project_likes("likes_reverse", "reverse")
        rows = self.client.write(
            """
            CALL gds.nodeSimilarity.write('likes_reverse', {
                nodeLabels: ['Movie', 'User'],
                topK: 10,
                similarityCutoff: 0.1,
                degreeCutoff: 5,
                writeRelationshipType: 'SIMILAR',
                writeProperty: 'score'
            })
            YIELD nodesCompared, relationshipsWritten
            RETURN nodesCompared, relationshipsWritten
            """
        )
        self._drop("likes_reverse")
        return rows[0]

    def louvain_and_embeddings(self) -> dict[str, Any]:
        """Community detection and FastRP embeddings on the undirected likes graph."""
        projection = self.project_likes("likes_undirected", "undirected")
        louvain = self.client.write(
            """
            CALL gds.louvain.write('likes_undirected', {writeProperty: 'community', concurrency: 1})
            YIELD communityCount, modularity
            RETURN communityCount, modularity
            """
        )[0]
        fastrp = self.client.write(
            """
            CALL gds.fastRP.write('likes_undirected', {
                embeddingDimension: $dim, randomSeed: $seed, iterationWeights: [0.0, 1.0, 1.0], concurrency: 1,
                writeProperty: 'embedding'
            })
            YIELD nodePropertiesWritten
            RETURN nodePropertiesWritten
            """,
            {"dim": EMBEDDING_DIM, "seed": RANDOM_SEED},
        )[0]
        self._drop("likes_undirected")
        return {"projection": projection, "louvain": louvain, "fastrp": fastrp}

    def pagerank(self) -> dict[str, Any]:
        """PageRank over the SIMILAR network: movies many others point to as "similar" rank high."""
        self._drop("similar")
        self.client.write(
            "CALL gds.graph.project('similar', 'Movie', {SIMILAR: {properties: 'score'}}) YIELD graphName "
            "RETURN graphName"
        )
        rows = self.client.write(
            """
            CALL gds.pageRank.write('similar', {writeProperty: 'pagerank', relationshipWeightProperty: 'score'})
            YIELD nodePropertiesWritten, ranIterations
            RETURN nodePropertiesWritten, ranIterations
            """
        )
        self._drop("similar")
        return rows[0]

    def run_all(self) -> tuple[dict[str, Any], float]:
        """Run the full pipeline in dependency order. Returns (results, seconds)."""
        start = time.perf_counter()
        results = {"node_similarity": self.node_similarity()}
        results["pagerank"] = self.pagerank()
        results.update(self.louvain_and_embeddings())
        return results, time.perf_counter() - start

    def status(self) -> dict[str, Any]:
        """Report whether algorithm outputs are present in the graph."""
        return self.client.read(
            """
            RETURN COUNT { ()-[:SIMILAR]->() } AS similarRelationships,
                   COUNT { (m:Movie) WHERE m.pagerank IS NOT NULL } AS moviesWithPagerank,
                   COUNT { (u:User) WHERE u.community IS NOT NULL } AS usersWithCommunity,
                   COUNT { (n) WHERE n.embedding IS NOT NULL } AS nodesWithEmbedding
            """
        )[0]
