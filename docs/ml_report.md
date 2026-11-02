# Link-prediction evaluation

Task: predict whether a user will like (rate >= 4) a movie, using FastRP graph embeddings.
Split: per user, their most recent 20% of likes are hidden from the graph and used as the test set.
Negatives: one random movie the user never rated per positive.

| Metric | FastRP + logistic regression | Popularity baseline |
|---|---:|---:|
| ROC AUC | 0.8294 | 0.8195 |
| Average precision | 0.8353 | 0.8095 |

Training likes: 39,105. Test likes: 8,363 (1,112 test likes dropped because the movie had no like in the training graph — the cold-start problem, which graph embeddings cannot solve).

Caveat: random negatives are mostly obscure movies, which flatters popularity; the baseline is therefore strong, and beating it is the meaningful test.

```json
{
  "split": "per-user leave-last-20%-out",
  "train_positive_edges": 39105,
  "test_positive_edges": 8363,
  "test_dropped_cold_start": 1112,
  "model_auc": 0.8294,
  "model_average_precision": 0.8353,
  "popularity_auc": 0.8195,
  "popularity_average_precision": 0.8095
}
```
