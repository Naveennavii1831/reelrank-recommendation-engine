\# ReelRank: Limitations and Responsible Use



\## 1. Project Purpose



ReelRank is an educational movie recommendation prototype developed to demonstrate a large-scale recommendation pipeline using machine learning, vector similarity search, an API, caching, and a frontend.



It is not yet a production-validated recommendation service.



\## 2. Dataset Limitations



The system uses historical MovieLens ratings. These ratings do not necessarily represent a user's current preferences.



Historical interactions can reflect exposure, popularity, and dataset collection patterns rather than a user's complete interests.



The model's recommendations are limited to the users, movies, and features represented in its training artifacts.



\## 3. Retrieval Limitations



The two-tower model generates user and movie embeddings, and FAISS retrieves similar movie candidates.



The current index contains 10,027 movies.



A relevant movie that is not retrieved cannot be recovered by the ranking model. Improving candidate coverage is therefore important.



\## 4. Ranking Limitations



The DCN-style ranking model estimates relevance from historical interactions.



Ratings of 4 or above are treated as positive examples in the recorded training setup. This binary transformation loses information contained in the original rating scale.



The recorded internal validation AUC is approximately 0.714. The difference between recorded training and validation losses suggests that additional overfitting analysis is warranted.



Model scores are not calibrated probabilities of a user watching or liking a movie.



\## 5. Evaluation Limitations



The recorded ranking comparison used a chronological validation cohort of 500 selected users.



Of those users, 475 had retrieved candidates and 151 had at least one relevant item among their candidates.



The absolute Recall@K and NDCG@K values remain low.



The comparison was conducted offline. It does not establish improvements in real-world engagement, satisfaction, retention, or revenue.



The project has no verified online impression and click dataset for calculating actual click-through rate.



\## 6. Cold-Start Limitations



New users may have insufficient interaction history for reliable personalized recommendations.



New movies may be missing from the trained embedding artifacts or FAISS index.



Popularity-based or content-based fallback recommendations could help address these cases, but their behavior must be implemented and tested before claiming complete cold-start support.



\## 7. Bias and Fairness



Historical rating data may contain popularity and exposure biases.



The model may favor movies with stronger historical representation or interactions.



Further work should examine recommendation diversity, catalog coverage, performance across user groups where appropriate, and the effect of popularity bias.



The project should not claim that its recommendations are unbiased without dedicated evaluation.



\## 8. Privacy and Security



The current project is intended for local educational use.



Before public deployment, the service should receive a security review, including:



\* Authentication and authorization.

\* Request validation and rate limiting.

\* Secure configuration and secret management.

\* Appropriate logging without unnecessary personal data.

\* Dependency and container security checks.

\* Monitoring and error handling.



\## 9. Performance and Reliability



The project includes Redis caching and Docker configuration, but this does not by itself establish production readiness.



Load testing, latency measurement, concurrency testing, failure recovery, and resource monitoring remain necessary.



Cache expiration and invalidation behavior should also be tested under realistic workloads.



\## 10. Responsible Use



Recommendations should be presented as algorithmic suggestions, not objective statements about a user's preferences.



Users should be able to understand that recommendations are generated from historical data and may not always be relevant.



Any production deployment should include suitable privacy, security, monitoring, and evaluation practices.



\## 11. Future Work



Recommended improvements include:



1\. Improve candidate retrieval recall.

2\. Tune the retrieval and ranking models.

3\. Compare against popularity and other baselines using matched evaluation cohorts.

4\. Evaluate recommendation diversity and catalog coverage.

5\. Implement and test cold-start fallbacks.

6\. Add automated tests and load testing.

7\. Collect appropriate impression and click data before conducting an online A/B test.

8\. Complete a security and deployment review.



\## 12. Conclusion



ReelRank demonstrates the main components of a two-stage personalized recommendation system. However, its recorded offline results, data limitations, and unverified production characteristics mean it should remain an educational prototype until further validation is completed.
