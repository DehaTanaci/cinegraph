"""Generate demo.ipynb (run `python -m scripts.build_notebook`, then execute it with nbconvert)."""

import nbformat as nbf

CELLS = [
    ("md", """# CineGraph — demonstration notebook

Graph exploration, graph algorithms, recommendations, the ML model and the guarded NL-to-Cypher assistant.

**Prerequisites:** `make docker-run && make seed && make gds && make train` (the API must answer on
`http://localhost:8080`, Ollama must be running for the assistant section)."""),
    ("code", """import json, os
import httpx
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from app.database.neo4j_client import get_client

os.makedirs("docs/img", exist_ok=True)
db = get_client()
API = "http://localhost:8080"
q = lambda cypher, **p: pd.DataFrame(db.read(cypher, p))"""),
    ("md", "## 1. What is in the graph"),
    ("code", """labels = q("MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS nodes ORDER BY nodes DESC")
rels = q("MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS count ORDER BY count DESC")
display(labels); display(rels)"""),
    ("md", """## 2. Rating behaviour

Ratings are skewed towards 3–4 stars, so we treat **≥ 4 as "liked"**. Activity is long-tailed: a few users rate
thousands of movies — the reason recommendation queries keep only the *top-30* most similar peers."""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
dist = q("MATCH ()-[r:RATED]->() RETURN r.rating AS rating, count(*) AS n ORDER BY rating")
axes[0].bar(dist.rating.astype(str), dist.n, color="#4C72B0"); axes[0].set_title("Rating values")
per_user = q("MATCH (u:User) RETURN u.ratingCount AS ratings")
axes[1].hist(per_user.ratings, bins=60, color="#DD8452", log=True); axes[1].set_title("Ratings per user (log count)")
plt.tight_layout(); plt.savefig("docs/img/ratings.png", dpi=110); plt.show()"""),
    ("code", """genres = q('''MATCH (g:Genre)<-[:IN_GENRE]-(m:Movie)
             RETURN g.name AS genre, count(m) AS movies, round(avg(m.avgRating), 2) AS avgRating ORDER BY movies DESC''')
ax = genres.plot.barh(x="genre", y="movies", figsize=(8, 6), legend=False, color="#55A868")
ax.invert_yaxis(); ax.set_title("Movies per genre (Genre nodes)"); plt.tight_layout()
plt.savefig("docs/img/genres.png", dpi=110); plt.show()"""),
    ("md", """## 3. Graph algorithms (GDS)

**Louvain** found taste communities on the users-like-movies graph. Below: community sizes and the genres that
over-index in each one compared with the whole catalogue (lift > 1 = more common than average)."""),
    ("code", """communities = q('''MATCH (m:Movie) WHERE m.community IS NOT NULL
   WITH m.community AS community, count(*) AS movies,
        COUNT { MATCH (u:User) WHERE u.community = m.community RETURN u } AS users
   RETURN community, movies ORDER BY movies DESC LIMIT 8''')
lift = q('''MATCH (m:Movie) WHERE m.community IS NOT NULL UNWIND m.genres AS g
   WITH m.community AS c, g, count(*) AS n
   CALL (c) { MATCH (x:Movie {community: c}) RETURN count(x) AS size }
   CALL (g) { MATCH (y:Movie) WHERE g IN y.genres RETURN count(y) AS gTotal }
   CALL () { MATCH (z:Movie) WHERE z.community IS NOT NULL RETURN count(z) AS total }
   WITH c, g, (toFloat(n) / size) / (toFloat(gTotal) / total) AS lift, n WHERE n >= 10
   ORDER BY lift DESC
   RETURN c AS community, collect(g + ' x' + toString(round(lift, 1)))[..3] AS overRepresentedGenres''')
communities.merge(lift, on="community")"""),
    ("md", "**Node Similarity** wrote `SIMILAR` edges. Toy Story's neighbourhood (edge width = similarity):"),
    ("code", """edges = db.read('''MATCH (a:Movie {movieId: 1})-[s:SIMILAR]->(b:Movie)
   OPTIONAL MATCH (b)-[s2:SIMILAR]->(c:Movie) WHERE (a)-[:SIMILAR]->(c)
   RETURN a.title AS a, b.title AS b, s.score AS w, c.title AS c, s2.score AS w2''')
G = nx.Graph()
for e in edges:
    G.add_edge(e["a"], e["b"], weight=e["w"])
    if e["c"]:
        G.add_edge(e["b"], e["c"], weight=e["w2"])
plt.figure(figsize=(11, 8)); pos = nx.spring_layout(G, seed=7, k=1.2)
nx.draw_networkx_edges(G, pos, width=[G[u][v]["weight"] * 12 for u, v in G.edges()], alpha=0.5)
nx.draw_networkx_nodes(G, pos, node_color=["#C44E52" if n == "Toy Story" else "#8172B3" for n in G], node_size=600)
nx.draw_networkx_labels(G, pos, font_size=8); plt.axis("off"); plt.title("SIMILAR neighbourhood of Toy Story")
plt.savefig("docs/img/toy_story_similar.png", dpi=110, bbox_inches="tight"); plt.show()"""),
    ("md", "**PageRank** on the SIMILAR network — movies that many others point to as \"similar\" (taste hubs):"),
    ("code", """hubs = httpx.get(f"{API}/api/movies/trending", params={"metric": "pagerank", "limit": 10}).json()["items"]
pd.DataFrame([{"title": i["title"], "year": i["year"], "pagerank": i["score"], "ratings": i["rating_count"]}
              for i in hubs])"""),
    ("md", "**Path-finding**: the strongest chain of similar movies from *Toy Story* to *The Godfather* (GDS Dijkstra):"),
    ("code", """httpx.get(f"{API}/api/movies/path", params={"from": 1, "to": 858, "weighted": True}).json()"""),
    ("md", "## 4. Recommendations and *why*"),
    ("code", """recs = httpx.get(f"{API}/api/users/1/recommendations", params={"limit": 5}).json()
pd.DataFrame([{"title": i["title"], "score": i["score"], "detail": i["detail"]} for i in recs["items"]])"""),
    ("code", """why = httpx.get(f"{API}/api/users/1/recommendations/{recs['items'][0]['movie_id']}/why").json()
print(why["summary"]); pd.DataFrame(why["evidence"])[["kind", "weight", "description"]].head(8)"""),
    ("code", """disc = httpx.get(f"{API}/api/users/1/discover", params={"limit": 5}).json()
pd.DataFrame([{"title": i["title"], "genres": i["genres"], "detail": i["detail"]} for i in disc["items"]])"""),
    ("md", "## 5. ML link prediction (FastRP embeddings + logistic regression)"),
    ("code", """from IPython.display import Markdown
display(Markdown(open("docs/ml_report.md").read()))
print(httpx.get(f"{API}/api/users/1/predict/589").json())"""),
    ("md", """## 6. NL-to-Cypher assistant and its guardrails

A successful question — note `rows` (verified data) vs `answer` (AI text, checked by `answer_grounded`):"""),
    ("code", """ok = httpx.post(f"{API}/api/ask", json={"question": "Best sci-fi movies of the 90s with at least 30 ratings?"},
                timeout=180).json()
print(ok["cypher"]); print(ok["answer"], "| grounded:", ok["answer_grounded"]); pd.DataFrame(ok["rows"]).head()"""),
    ("md", "Refused before the LLM is called (asks to modify data), and refused because the data does not exist:"),
    ("code", """for question in ["Delete all the movies from the database", "Which movies had the biggest box office revenue?"]:
    r = httpx.post(f"{API}/api/ask", json={"question": question}, timeout=180).json()
    print(f"{question!r} -> {r['status']}: {r['trace'][-1]}")"""),
    ("md", "What happens when the LLM *outputs* a dangerous or invented query — each is blocked before execution:"),
    ("code", """bad_outputs = {
    "write": "MATCH (m:Movie) DETACH DELETE m",
    "invented schema": "MATCH (m:Movie)-[:DIRECTED_BY]->(d:Director) RETURN d.name, m.boxOffice",
    "two statements": "MATCH (m:Movie) RETURN m.title; MATCH (n) DETACH DELETE n",
    "admin procedure": "CALL dbms.listConfig() YIELD name RETURN name",
    "cartesian blow-up": "MATCH (a:Movie), (b:Movie) RETURN a.title, b.title",
}
rows = []
for kind, cypher in bad_outputs.items():
    report = httpx.post(f"{API}/api/ask/check", json={"cypher": cypher}).json()
    failed = [f"{c['name']}: {c['detail']}" for c in report["checks"] if not c["passed"]]
    rows.append({"case": kind, "passed": report["passed"], "blocked by": failed[0] if failed else ""})
pd.DataFrame(rows)"""),
    ("code", """display(Markdown(open("docs/agent_evaluation.md").read().split("| Question |")[0]))"""),
    ("md", "## 7. Index usage — `EXPLAIN` and `PROFILE`"),
    ("code", """for cypher in ["MATCH (m:Movie {movieId: 1}) RETURN m.title",
               "MATCH (m:Movie) WHERE m.year >= 1990 AND m.year <= 1999 RETURN count(m)",
               "MATCH ()-[r:RATED]->() WHERE r.timestamp >= datetime('2018-06-24') RETURN count(r)"]:
    plan = db.explain(cypher); prof = db.profile(cypher)
    print(f"{cypher}\\n   operators: {[o.split('@')[0] for o in plan['operators']]}  db hits: {prof['db_hits']:,}\\n")"""),
    ("code", """display(Markdown(open("docs/benchmarks.md").read()))"""),
]


def main() -> None:
    """Write demo.ipynb."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                      for kind, src in CELLS]
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(notebook, "demo.ipynb")


if __name__ == "__main__":
    main()
