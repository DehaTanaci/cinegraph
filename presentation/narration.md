# CineGraph demo video — narration script

Read each section while the matching footage plays. Timestamps are where each scene starts in `CineGraph_demo.mp4`.

## 1. Intro — 0:00

- Hi, I'm Deha Tanaci. This is CineGraph, my knowledge graph project for the AIDAMS GraphDB course: a movie recommender built on Neo4j, with graph algorithms, a machine learning model, and a guarded natural-language assistant running on a local language model.
- Everything in this video is the real, running project. Recommendation is a question about connections: people who liked what you liked also liked something you haven't seen. In a relational database that's three self-joins on one ratings table; in a graph it's a single pattern, and its cost depends on your neighbourhood, not on the size of the whole database.

## 2. Terminal Stack — 0:36

- Let's start from the terminal. The Makefile wraps every command of the project. *(on screen: action)*
- The whole system runs with Docker Compose: three services, all healthy. Neo4j 5 with the Graph Data Science plugin, the FastAPI application, and an Nginx reverse proxy, which is the only public entry point. *(on screen: action)*
- The language model, Qwen 2.5 Coder with seven billion parameters, runs locally in Ollama: no paid API, and no data leaves the machine. *(on screen: action)*

## 3. Neo4J Schema — 1:05

- This is the database itself, in the Neo4j Browser. Here is the schema of the graph. *(on screen: action)*
- Four node labels: User, Movie, Genre and Tag. Users rate movies; movies belong to genres and carry tags; and movies are linked to each other by SIMILAR, which is computed by a graph algorithm.
- These were deliberate choices. A rating is a relationship, not a node: it always connects exactly one user and one movie, and nothing else points at it, so a node would only add a hop to every traversal.
- Genres are stored twice: as nodes, so we can walk from a movie to its genre to other movies, and as a list on each movie, so a filter like 'only sci-fi' needs no extra hop. A test checks the two never disagree. And 'liked' means a rating of four or more, applied at query time.
- Here is Toy Story and its most similar movies. *(on screen: action)*
- Toy Story 2, the Star Wars trilogy, Jurassic Park, The Lion King. No genre data was used: these links come purely from who liked what.

## 4. Readme Data — 2:04

- The README documents the whole project. The data is MovieLens, from GroupLens: real ratings, about a hundred thousand of them, from six hundred and ten users on almost ten thousand movies. *(on screen: action)*
- The loading script cleans it on the way in: it parses the year out of titles, restores articles like 'Matrix, The', normalises tags, and drops the fake 'no genres listed' genre.
- I also asked an AI assistant to design a schema for the same project, and compared it with mine. It invented user names, emails and ages that the dataset doesn't contain, and stored every rating twice. This appendix explains each difference. *(on screen: action)*

## 5. Demo Foryou — 2:42

- Now the demo interface. I pick user one. *(on screen: action)*
- The profile comes from the graph: two hundred and thirty-two ratings, and a clear taste for action and adventure.
- The recommendations come from collaborative filtering: find the thirty users whose likes overlap most with yours, then collect what they loved that you haven't seen. Terminator 2 comes first.
- And the key question: why was this recommended? *(on screen: action)*
- The explanation is built only from real paths in the graph: similar users who loved it, its similarity to The Terminator, which this user rated five stars, and their taste for action. No language model writes this, so it can't invent a reason.

## 6. Demo Movies — 3:20

- On the right, discovery. Louvain community detection groups users and movies into taste communities. *(on screen: action)*
- These are movies from communities this user rarely explores, but that are popular in their own community.
- In the Movies tab, I search for The Matrix. *(on screen: action)*
- On the left, movies liked by the same people, computed by node similarity: Fight Club, Saving Private Ryan, The Empire Strikes Back.
- On the right, movies that only share genres and tags. Two very different notions of similar, and the graph supports both.

## 7. Demo Path — 3:52

- Taste path finds the chain of similar movies linking any two films.
- From Toy Story to The Godfather, Dijkstra's algorithm runs on the similarity network, with a cost of one minus similarity, so it prefers strong links. *(on screen: action)*
- Toy Story, Star Wars, The Godfather: two hops.

## 8. Readme Gds — 4:09

- All the algorithms run in Neo4j's Graph Data Science library, on in-memory projections, and their results are written back to the graph, so requests only read them. *(on screen: action)*
- Node Similarity creates the SIMILAR links, PageRank finds taste hubs, Louvain finds communities, and FastRP turns every user and movie into a sixty-four-dimensional embedding.
- Those embeddings feed a machine learning model that predicts whether a user will like a movie. To evaluate it honestly, each user's most recent twenty percent of likes are hidden. *(on screen: action)*
- The model reaches an AUC of 0.83, against 0.82 for simply guessing popular movies: a modest win. My first version actually lost to that baseline because of a bad time split, and I kept that finding in the report.

## 9. Demo Ask — 4:55

- Now the assistant. I ask: what are the best sci-fi movies of the 90s, with at least thirty ratings? *(on screen: action)*
- The local model writes a Cypher query. Before it runs, it passes every guardrail listed on the right: one statement, no write keywords, every label and property must exist in the schema, a row limit, and Neo4j's own EXPLAIN must classify it as read-only.
- The table is verified data from the database. The sentence on top is AI-generated, so it's checked too: every number it mentions must appear in the rows.
- Now a question the data can't answer: who directed Pulp Fiction? *(on screen: action)*
- There is no director data, so instead of substituting something else, it declines.
- A request to delete all movies is refused before the model is even called. *(on screen: action)*
- And here is what happens when the model itself produces a dangerous query: a delete, using a relationship that doesn't exist. *(on screen: action)*
- It's blocked by three separate checks, and never reaches the database. Even if every check failed, queries run in a read-only transaction with a timeout, so Neo4j itself would refuse the write.

## 10. Readme Eval — 6:00

- The principle is that the language model only proposes text; deterministic code and Neo4j decide what runs and what counts as a fact. *(on screen: action)*
- On a twenty-question evaluation, it answered all fourteen answerable questions correctly and refused all six unsafe or unanswerable ones. The first run scored nineteen: the model answered a box-office question with something else, so I added an explicit 'data not in the graph' path. *(on screen: action)*

## 11. Swagger — 6:25

- Behind the demo page is a FastAPI service. Swagger documents every endpoint, and every input and output is validated by Pydantic models. *(on screen: action)*
- Here I call the recommendations endpoint directly. Errors use meaningful codes: four oh four for an unknown user, four twenty-two for invalid input, and four oh one for the admin endpoint without its API key. *(on screen: action)*

## 12. Code — 6:47

- A quick look at the code. This is the recommendation query. It starts from one user, found through a uniqueness constraint, expands to peers through shared likes, and keeps only the top thirty.
- A NOT EXISTS subquery guarantees you're never recommended something you've already seen, the genre filter uses the list property, and every value is a parameter.
- And this is the LangGraph workflow: screen the question, generate, validate, execute, and summarise, with failed validations fed back to the model for up to three attempts. *(on screen: action)*

## 13. Readme Perf — 7:18

- Every optimisation was measured with PROFILE, before and after. *(on screen: action)*
- The index on rating timestamps cuts the work for recent trending by thirty-four times. Limiting recommendations to the top thirty peers cuts it by almost four, from about two hundred milliseconds to fifty. Precomputing average ratings makes the top-rated list almost three hundred times cheaper.

## 14. Terminal Quality — 7:40

- Code quality: Pylint with the course configuration, above the required nine point five. *(on screen: action)*
- And the test suite: ninety-six tests, from unit tests of every guardrail to Cypher tests against the real database and health checks of the running containers. Coverage is eighty-three percent. *(on screen: action)*

## 15. Github — 8:02

- Each part of the project was merged through a reviewed pull request, and the reviews caught real bugs that were fixed before merging.
- GitHub Actions runs the linter and the full test suite against a real Neo4j on every push and pull request. *(on screen: action)*
- And the API image is published on Docker Hub, for both Intel and ARM machines. *(on screen: action)*

## 16. Outro — 8:25

- To wrap up, the limitations: new users with no ratings only get popularity-based lists, and the local model is much weaker than a hosted one, which is exactly why nothing it writes is trusted. *(on screen: action)*
- Everything is public on GitHub and reproducible with a handful of make commands. Thanks for watching.
