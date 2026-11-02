"""Graph queries behind the movie and user endpoints.

Every query is parameterised (never string-formatted with user input), bounded with LIMIT,
and anchored on an indexed lookup (movieId / userId) so cost depends on the neighbourhood
of one node, not on the size of the whole graph.
"""

from typing import Any, Optional

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient

# A taste community counts as one of the user's "home" communities if they liked at least this many of its movies.
HOME_COMMUNITY_MIN_LIKES = 3

MOVIE_FIELDS = """
    movie_id: {v}.movieId, title: {v}.title, year: {v}.year, genres: {v}.genres,
    avg_rating: {v}.avgRating, rating_count: {v}.ratingCount
"""


def movie_map(var: str) -> str:
    """Cypher map projection of the standard movie fields for variable `var`."""
    return "{" + MOVIE_FIELDS.format(v=var) + "}"


class NotFoundError(Exception):
    """Raised when a requested movie or user does not exist."""


class PrerequisiteError(Exception):
    """Raised when an endpoint needs graph-algorithm output that has not been computed yet."""


class MovieService:
    """Read-side graph queries for movies and users."""

    def __init__(self, client: Neo4jClient):
        self.client = client
        self.like = get_settings().like_threshold

    # ---------- lookups ----------

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        """One movie with its most-used tags."""
        rows = self.client.read(
            f"""
            MATCH (m:Movie {{movieId: $id}})
            OPTIONAL MATCH (m)-[t:TAGGED_AS]->(tag:Tag)
            WITH m, tag ORDER BY t.count DESC
            WITH m, collect(tag.name)[..8] AS tags
            RETURN {{{MOVIE_FIELDS.format(v='m')}, imdb_id: m.imdbId, tmdb_id: m.tmdbId, top_tags: tags,
                   pagerank: m.pagerank, community: m.community}} AS movie
            """,
            {"id": movie_id},
        )
        if not rows:
            raise NotFoundError(f"Movie {movie_id} not found")
        return rows[0]["movie"]

    def search_movies(self, text: str, limit: int, offset: int) -> list[dict[str, Any]]:
        """Full-text title search (uses the movie_title_ft index)."""
        escaped = "".join("\\" + c if c in '+-&|!(){}[]^"~*?:\\/' else c for c in text)
        return self.client.read(
            f"""
            CALL db.index.fulltext.queryNodes('movie_title_ft', $q) YIELD node AS m, score
            RETURN {movie_map('m')} AS movie, score
            ORDER BY score DESC, m.ratingCount DESC
            SKIP $offset LIMIT $limit
            """,
            {"q": escaped + "~" if " " not in escaped else escaped, "limit": limit, "offset": offset},
        )

    def get_user(self, user_id: int) -> dict[str, Any]:
        """Taste profile: rating stats and favourite genres (by number of liked movies)."""
        rows = self.client.read(
            """
            MATCH (u:User {userId: $id})
            CALL (u) {
                MATCH (u)-[r:RATED]->()
                RETURN avg(r.rating) AS avgRating
            }
            CALL (u) {
                MATCH (u)-[r:RATED]->(:Movie)-[:IN_GENRE]->(g:Genre)
                WHERE r.rating >= $like
                WITH g, count(*) AS liked
                ORDER BY liked DESC LIMIT 5
                RETURN collect({genre: g.name, liked_movies: liked}) AS favourites
            }
            RETURN u.userId AS user_id, u.ratingCount AS rating_count, round(avgRating, 2) AS avg_rating,
                   favourites AS favourite_genres, u.community AS community
            """,
            {"id": user_id, "like": self.like},
        )
        if not rows:
            raise NotFoundError(f"User {user_id} not found")
        return rows[0]

    def _require_user(self, user_id: int) -> None:
        if not self.client.read("MATCH (u:User {userId: $id}) RETURN 1 AS ok", {"id": user_id}):
            raise NotFoundError(f"User {user_id} not found")

    def _require_movie(self, movie_id: int) -> None:
        if not self.client.read("MATCH (m:Movie {movieId: $id}) RETURN 1 AS ok", {"id": movie_id}):
            raise NotFoundError(f"Movie {movie_id} not found")

    # ---------- similar movies ----------

    def similar_movies(self, movie_id: int, method: str, limit: int, offset: int) -> list[dict[str, Any]]:
        """Movies similar to `movie_id`.

        method="collaborative": precomputed GDS Node Similarity (people who liked A also liked B).
        method="content": live weighted Jaccard over shared genres and tags.
        """
        self._require_movie(movie_id)
        if method == "collaborative":
            rows = self.client.read(
                f"""
                MATCH (:Movie {{movieId: $id}})-[s:SIMILAR]->(o:Movie)
                RETURN {movie_map('o')} AS movie, round(s.score, 4) AS score,
                       'share ' + toString(round(s.score * 100)) + '% of their fans' AS detail
                ORDER BY score DESC SKIP $offset LIMIT $limit
                """,
                {"id": movie_id, "limit": limit, "offset": offset},
            )
            if not rows and offset == 0 and not self.client.read("MATCH ()-[:SIMILAR]->() RETURN 1 AS x LIMIT 1"):
                raise PrerequisiteError("Similarity has not been computed yet. Run `make gds`.")
            return rows
        return self.client.read(
            f"""
            MATCH (m:Movie {{movieId: $id}})
            WITH m, COUNT {{ (m)-[:IN_GENRE|TAGGED_AS]->() }} AS mFeatures
            MATCH (m)-[:IN_GENRE|TAGGED_AS]->(f)<-[:IN_GENRE|TAGGED_AS]-(o:Movie)
            WHERE o <> m AND o.ratingCount >= 10
            WITH m, mFeatures, o, count(f) AS shared, collect(coalesce(f.name, ''))[..5] AS sharedNames
            WITH o, shared, sharedNames,
                 toFloat(shared) / (mFeatures + COUNT {{ (o)-[:IN_GENRE|TAGGED_AS]->() }} - shared) AS jaccard
            RETURN {movie_map('o')} AS movie, round(jaccard, 4) AS score,
                   'shares: ' + reduce(s = '', n IN sharedNames | s + CASE s WHEN '' THEN '' ELSE ', ' END + n)
                   AS detail
            ORDER BY score DESC, o.ratingCount DESC SKIP $offset LIMIT $limit
            """,
            {"id": movie_id, "limit": limit, "offset": offset},
        )

    # ---------- recommendations ----------

    def recommendations(
        self, user_id: int, limit: int, offset: int, genre: Optional[str] = None, peers: int = 30
    ) -> list[dict[str, Any]]:
        """User-based collaborative filtering over the graph.

        1. find the `peers` users who liked the most movies in common with this user;
        2. collect movies those peers liked that this user has not rated;
        3. score each movie by the summed overlap of the peers who liked it.
        The genre filter uses the `genres` list property: no extra hop to :Genre needed.
        """
        self._require_user(user_id)
        return self.client.read(
            f"""
            MATCH (u:User {{userId: $id}})-[r1:RATED]->(:Movie)<-[r2:RATED]-(peer:User)
            WHERE r1.rating >= $like AND r2.rating >= $like AND peer <> u
            WITH u, peer, count(*) AS overlap
            ORDER BY overlap DESC LIMIT $peers
            MATCH (peer)-[r:RATED]->(rec:Movie)
            WHERE r.rating >= $like
              AND NOT EXISTS {{ (u)-[:RATED]->(rec) }}
              AND ($genre IS NULL OR $genre IN rec.genres)
            WITH rec, sum(overlap) AS score, count(peer) AS supporters
            RETURN {movie_map('rec')} AS movie, toFloat(score) AS score,
                   toString(supporters) + ' users with similar taste loved it' AS detail
            ORDER BY score DESC, rec.avgRating DESC SKIP $offset LIMIT $limit
            """,
            {"id": user_id, "like": self.like, "peers": peers, "genre": genre, "limit": limit, "offset": offset},
        )

    def why_recommended(self, user_id: int, movie_id: int) -> dict[str, Any]:
        """Collect the concrete graph paths connecting a user to a movie."""
        self._require_user(user_id)
        self._require_movie(movie_id)
        params = {"uid": user_id, "mid": movie_id, "like": self.like}
        rated = self.client.read(
            "MATCH (:User {userId: $uid})-[r:RATED]->(:Movie {movieId: $mid}) RETURN r.rating AS rating", params
        )
        peers = self.client.read(
            """
            MATCH (u:User {userId: $uid})-[r1:RATED]->(shared:Movie)<-[r2:RATED]-(peer:User)
                  -[r3:RATED]->(rec:Movie {movieId: $mid})
            WHERE r1.rating >= $like AND r2.rating >= $like AND r3.rating >= $like AND peer <> u
            WITH peer, r3, count(shared) AS overlap, collect(shared.title)[..3] AS examples
            ORDER BY overlap DESC LIMIT 5
            RETURN peer.userId AS peer, overlap, examples, r3.rating AS peerRating
            """,
            params,
        )
        similar = self.client.read(
            """
            MATCH (u:User {userId: $uid})-[r:RATED]->(liked:Movie)-[s:SIMILAR]-(rec:Movie {movieId: $mid})
            WHERE r.rating >= $like
            RETURN DISTINCT liked.title AS liked, r.rating AS rating, round(s.score, 3) AS score
            ORDER BY score DESC LIMIT 5
            """,
            params,
        )
        genres = self.client.read(
            """
            MATCH (rec:Movie {movieId: $mid})-[:IN_GENRE]->(g:Genre)
            CALL (g) {
                MATCH (:User {userId: $uid})-[r:RATED]->(:Movie)-[:IN_GENRE]->(g)
                WHERE r.rating >= $like
                RETURN count(*) AS likedInGenre
            }
            WITH g, likedInGenre WHERE likedInGenre > 0
            RETURN g.name AS genre, likedInGenre ORDER BY likedInGenre DESC
            """,
            params,
        )
        tags = self.client.read(
            """
            MATCH (u:User {userId: $uid})-[r:RATED]->(liked:Movie)-[:TAGGED_AS]->(t:Tag)
                  <-[:TAGGED_AS]-(rec:Movie {movieId: $mid})
            WHERE r.rating >= $like AND liked <> rec
            WITH t, collect(DISTINCT liked.title)[..3] AS via, count(DISTINCT liked) AS n
            RETURN t.name AS tag, via, n ORDER BY n DESC LIMIT 5
            """,
            params,
        )
        return {
            "rated": rated[0]["rating"] if rated else None,
            "movie": self.client.read(f"MATCH (m:Movie {{movieId: $mid}}) RETURN {movie_map('m')} AS m", params)[0][
                "m"
            ],
            "peers": peers,
            "similar": similar,
            "genres": genres,
            "tags": tags,
        }

    def discover(self, user_id: int, limit: int, offset: int) -> list[dict[str, Any]]:
        """Movies from taste communities this user has barely explored, loved by their own community.

        "Home" communities are those where the user liked at least HOME_COMMUNITY_MIN_LIKES movies.
        """
        self._require_user(user_id)
        if not self.client.read("MATCH (u:User {userId: $id}) WHERE u.community IS NOT NULL RETURN 1 AS ok",
                                {"id": user_id}):
            raise PrerequisiteError("Communities have not been computed yet (or this user has no liked movies). "
                                    "Run `make gds`.")
        return self.client.read(
            f"""
            MATCH (u:User {{userId: $id}})
            CALL (u) {{
                MATCH (u)-[r:RATED]->(m:Movie)
                WHERE r.rating >= $like AND m.community IS NOT NULL
                WITH m.community AS c, count(*) AS n WHERE n >= $homeMin
                RETURN collect(c) AS home
            }}
            MATCH (peer:User {{community: u.community}})-[r:RATED]->(rec:Movie)
            WHERE r.rating >= $like AND peer <> u
              AND NOT rec.community IN home
              AND NOT EXISTS {{ (u)-[:RATED]->(rec) }}
            WITH rec, count(peer) AS fans
            RETURN {movie_map('rec')} AS movie, toFloat(fans) AS score,
                   'outside your usual taste clusters; ' + toString(fans) + ' people in your community loved it'
                   AS detail
            ORDER BY score DESC, rec.pagerank DESC SKIP $offset LIMIT $limit
            """,
            {"id": user_id, "like": self.like, "homeMin": HOME_COMMUNITY_MIN_LIKES, "limit": limit, "offset": offset},
        )

    # ---------- rankings ----------

    def trending(self, metric: str, window_days: int, min_ratings: int, limit: int, offset: int) -> list[dict]:
        """Ranked movies.

        metric="recent": most-loved in the last `window_days` before the newest rating in the data
                         (relative, because MovieLens is a historical snapshot). Uses the RATED timestamp index.
        metric="pagerank": centrality in the SIMILAR network (precomputed by GDS).
        metric="rating": highest average rating among movies with at least `min_ratings` ratings.
        """
        if metric == "pagerank":
            return self.client.read(
                f"""
                MATCH (m:Movie) WHERE m.pagerank IS NOT NULL AND m.ratingCount >= $min
                RETURN {movie_map('m')} AS movie, round(m.pagerank, 3) AS score, 'taste-network centrality' AS detail
                ORDER BY m.pagerank DESC SKIP $offset LIMIT $limit
                """,
                {"min": min_ratings, "limit": limit, "offset": offset},
            )
        if metric == "rating":
            return self.client.read(
                f"""
                MATCH (m:Movie) WHERE m.ratingCount >= $min
                RETURN {movie_map('m')} AS movie, m.avgRating AS score,
                       toString(m.ratingCount) + ' ratings' AS detail
                ORDER BY m.avgRating DESC, m.ratingCount DESC SKIP $offset LIMIT $limit
                """,
                {"min": min_ratings, "limit": limit, "offset": offset},
            )
        return self.client.read(
            f"""
            CALL () {{
                MATCH ()-[r:RATED]->() WHERE r.timestamp IS NOT NULL
                RETURN r.timestamp AS latest ORDER BY r.timestamp DESC LIMIT 1
            }}
            MATCH ()-[r:RATED]->(m:Movie)
            WHERE r.timestamp >= latest - duration({{days: $days}})
            WITH m, count(r) AS recent, avg(r.rating) AS recentAvg
            WHERE recent >= $min
            RETURN {movie_map('m')} AS movie, round(recent * recentAvg, 2) AS score,
                   toString(recent) + ' ratings in window, avg ' + toString(round(recentAvg, 2)) AS detail
            ORDER BY score DESC SKIP $offset LIMIT $limit
            """,
            {"days": window_days, "min": min_ratings, "limit": limit, "offset": offset},
        )

    # ---------- pathfinding ----------

    def taste_path(self, from_id: int, to_id: int, weighted: bool) -> dict[str, Any]:
        """Shortest chain of similar movies linking two movies.

        weighted=False: Cypher shortestPath (fewest hops), depth capped at 8.
        weighted=True: GDS Dijkstra with cost = 1 - similarity (prefers strong links).
        """
        self._require_movie(from_id)
        self._require_movie(to_id)
        params = {"a": from_id, "b": to_id}
        if not weighted:
            rows = self.client.read(
                """
                MATCH (a:Movie {movieId: $a}), (b:Movie {movieId: $b})
                MATCH p = shortestPath((a)-[:SIMILAR*..8]-(b))
                RETURN [n IN nodes(p) | {movie_id: n.movieId, title: n.title}] AS steps,
                       [r IN relationships(p) | r.score] AS scores
                """,
                params,
            )
            if not rows:
                return {"method": "shortestPath", "hops": -1, "steps": []}
            steps, scores = rows[0]["steps"], rows[0]["scores"]
            for i, step in enumerate(steps):
                step["similarity_to_next"] = round(scores[i], 4) if i < len(scores) else None
            return {"method": "shortestPath", "hops": len(scores), "steps": steps}

        self.client.write("CALL gds.graph.drop('similar_cost', false) YIELD graphName RETURN graphName")
        self.client.write(
            """
            MATCH (s:Movie)-[r:SIMILAR]->(t:Movie)
            WITH gds.graph.project('similar_cost', s, t, {relationshipProperties: {cost: 1.0 - r.score}},
                                   {undirectedRelationshipTypes: ['*']}) AS g
            RETURN g.nodeCount AS n
            """
        )
        try:
            rows = self.client.read(
                """
                MATCH (a:Movie {movieId: $a}), (b:Movie {movieId: $b})
                CALL gds.shortestPath.dijkstra.stream('similar_cost', {
                    sourceNode: a, targetNode: b, relationshipWeightProperty: 'cost'
                })
                YIELD totalCost, nodeIds, costs
                RETURN totalCost,
                       [i IN range(0, size(nodeIds) - 1) |
                         {movie_id: gds.util.asNode(nodeIds[i]).movieId, title: gds.util.asNode(nodeIds[i]).title,
                          similarity_to_next: CASE WHEN i < size(nodeIds) - 1
                                              THEN round(1.0 - (costs[i + 1] - costs[i]), 4) END}] AS steps
                """,
                params,
            )
        finally:
            self.client.write("CALL gds.graph.drop('similar_cost', false) YIELD graphName RETURN graphName")
        if not rows:
            return {"method": "dijkstra", "hops": -1, "steps": []}
        return {
            "method": "dijkstra",
            "hops": len(rows[0]["steps"]) - 1,
            "total_cost": round(rows[0]["totalCost"], 4),
            "steps": rows[0]["steps"],
        }
