\# ReelRank Model Evaluation



\## 1. Objective



This document records the offline evaluation of the ReelRank recommendation pipeline.



The system uses two stages:



1\. FAISS-based candidate retrieval.

2\. DCN-based candidate reranking.



The purpose of the comparison is to measure whether reranking improves the ordering of retrieved movies.



\## 2. Dataset and Evaluation Setup



The project uses MovieLens 25M data with a chronological global 80/10/10 training, validation, and test split.



The ranking evaluation used the chronological validation partition.



Recorded evaluation settings:



\* Selected users: 500

\* Users with retrieved candidates: 475

\* Users with at least one relevant item among retrieved candidates: 151

\* Requested candidate count: 100

\* Positive relevance threshold: rating ≥ 4.0



The ranking evaluation is candidate-constrained: the ranking model can only reorder movies supplied by retrieval.



\## 3. Evaluation Metrics



\*\*Recall@K\*\* measures the proportion of relevant items recovered within the top K recommendations, according to the evaluation's relevance definition.



\*\*NDCG@K\*\* measures ranking quality while giving greater weight to relevant items appearing near the top of the recommendation list.



The metrics below are offline ranking metrics. They do not directly measure click-through rate, revenue, user satisfaction, or business impact.



\## 4. Recorded Results



| Metric    | FAISS retrieval | DCN reranking |

| --------- | --------------: | ------------: |

| Recall@10 |        0.002489 |      0.007033 |

| Recall@20 |        0.004699 |      0.010783 |

| NDCG@10   |        0.006752 |      0.012125 |

| NDCG@20   |        0.007448 |      0.012816 |



The recorded results show improved metrics after DCN reranking on the evaluated cohort.



\## 5. Relative Improvement



| Metric    | Reported relative improvement |

| --------- | ----------------------------: |

| Recall@10 |                       182.55% |

| Recall@20 |                       129.46% |

| NDCG@10   |                        79.58% |

| NDCG@20   |                        72.07% |



Relative improvements should be interpreted alongside absolute metric values. A large relative improvement does not necessarily mean that overall recommendation quality is high.



\## 6. Paired Bootstrap Analysis



The recorded paired bootstrap analysis for NDCG@10 reported:



\* Mean difference: approximately 0.005373

\* 95% confidence interval: approximately 0.0010965 to 0.0099813

\* Bootstrap samples: 2,000

\* Random seed: 42



The interval is above zero in this recorded experiment. This supports a positive offline difference for the evaluated sample and methodology; it does not establish online business impact or guarantee the same result on other cohorts.



\## 7. Training Results



\### Retrieval model



The two-tower retrieval model was trained for five epochs using a 500,000-interaction sample.



Recorded details:



\* Embedding dimension: 32

\* Indexed movies: 10,027

\* User embeddings: 85,244

\* Training duration: approximately 45.38 seconds

\* Final recorded training loss: approximately 8247.31



The loss value is specific to the implemented objective and should not be compared directly with the ranking model's loss.



\### Ranking model



The DCN-style ranking model was trained for five epochs using a 300,000-interaction sample.



Recorded details:



\* Positive label rate: approximately 0.49843

\* Validation AUC: approximately 0.71396

\* Validation log loss: approximately 0.88001

\* Final recorded training loss: approximately 0.2958

\* Training duration: approximately 32.08 seconds



The difference between training loss and validation loss suggests possible overfitting. Further analysis and tuning are needed.



\## 8. Interpretation



The recorded offline comparison indicates that DCN reranking improved Recall@K and NDCG@K compared with the original retrieval order on the selected validation cohort.



However:



\* Absolute ranking metrics remain low.

\* Some users had no retrieved candidates.

\* Only 151 of the 500 selected users had relevant items among their candidates.

\* The experiment evaluates reranking on retrieved candidates, not all possible movies.

\* The retrieval and ranking stages need further optimization.



\## 9. Experiment Limitations



This was an offline paired comparison, not a live online A/B test.



The project does not have the impression and click logs required to calculate actual CTR. Consequently, no online CTR improvement is claimed.



Results depend on the data split, user cohort, candidate-generation method, relevance threshold, and implementation.



The earlier popularity-baseline evaluation used a different cohort and should not be compared directly with these ranking results as if the experiments were matched.



\## 10. Reproducibility



The recorded artifacts include:



\* `reports/retrieval\_training\_report.json`

\* `reports/ranking\_training\_report.json`

\* `reports/ranking\_evaluation.json`

\* `reports/ab\_test\_results.json`



Use these reports as the source of truth when checking the recorded results.



\## 11. Conclusion



On the recorded chronological validation experiment, DCN reranking improved the reported Recall@10, Recall@20, NDCG@10, and NDCG@20 metrics over FAISS retrieval ordering.



The result is promising for this educational prototype, but further work is required to improve absolute recommendation quality, validate generalization, assess cold-start behavior, and test operational performance.
