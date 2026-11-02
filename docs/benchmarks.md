# Query benchmarks (before / after)

Dataset: `ml-latest-small`. Median of 7 runs after one warm-up; db hits from `PROFILE`.

## Indexes

| Case | db hits before | db hits after | Less work | Latency ms | Plan |
|---|---:|---:|---:|---:|---|
| Movies of a decade (range filter on year) | 21,697 | 4,425 | **4.9×** | 33.8 → 30.3 | `NodeByLabelScan` → `NodeIndexSeekByRange` |
| Ratings in the last 90 days (relationship index on RATED.timestamp) | 273,829 | 7,982 | **34.3×** | 67.3 → 26.6 | `NodeByLabelScan` → `DirectedRelationshipIndexSeekByRange` |
| User lookup by id (uniqueness constraint = index) | 12,015 | 10,796 | **1.1×** | 36.4 → 35.5 | `NodeByLabelScan` → `NodeUniqueIndexSeek` |

## Query rewrites

| Case | db hits before | db hits after | Less work | Latency ms | Plan |
|---|---:|---:|---:|---:|---|
| Recommendations: all peers vs. top-30 peers | 2,236,063 | 599,547 | **3.7×** | 215.7 → 52.5 | `NodeUniqueIndexSeek` → `NodeUniqueIndexSeek` |
| Movie list: aggregate ratings per request vs. precomputed avgRating | 262,101 | 911 | **287.7×** | 26.2 → 3.2 | `NodeByLabelScan` → `NodeIndexSeekByRange` |
| Two movies: Cartesian product vs. index-anchored pattern | 39,002 | 36 | **1,083.4×** | 10.0 → 2.4 | `CartesianProduct, NodeByLabelScan` → `CartesianProduct, NodeUniqueIndexSeek` |
