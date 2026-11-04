\# ReelRank System Architecture



\## 1. Overview



ReelRank uses a two-stage recommendation architecture to generate personalized movie recommendations from historical user ratings.



The two primary machine-learning components are:



1\. \*\*Candidate retrieval:\*\* A two-tower model generates user and movie embeddings. FAISS searches the movie embedding index to retrieve candidate movies.

2\. \*\*Candidate ranking:\*\* A Deep \& Cross Network (DCN)-style model scores the retrieved candidates and reorders them.



FastAPI exposes the recommendation functionality, Redis caches responses, and a React application provides the user interface.



\## 2. Data Pipeline



The project uses MovieLens 25M interaction data.



The preprocessing pipeline creates chronological training, validation, and test partitions. User and movie feature tables are derived from the training partition.



Important artifacts include:



\* `data/processed/train.csv`

\* `data/processed/validation.csv`

\* `data/processed/test.csv`

\* `data/processed/user\_features.csv`

\* `data/processed/movie\_features.csv`

\* `data/processed/movies\_clean.csv`



A sampled interaction dataset is used to make model training more manageable.



\## 3. Candidate Retrieval



The retrieval component uses a two-tower architecture:



\* The user tower produces a 32-dimensional user embedding.

\* The movie tower produces a 32-dimensional movie embedding.

\* FAISS performs similarity search over the indexed movie embeddings.



The generated artifacts include:



\* `models/two\_tower.weights.h5`

\* `models/retrieval\_embeddings.npz`

\* `models/retrieval\_metadata.json`

\* `models/movie\_faiss.index`

\* `models/faiss\_movie\_ids.npy`



The current FAISS index contains 10,027 movie embeddings.



\## 4. Candidate Ranking



The DCN-style ranking model scores candidate movies using user and movie features.



The model is trained as a binary relevance predictor, with ratings of 4 or above treated as positive examples.



The ranking artifacts include:



\* `models/dcn\_ranker.keras`

\* `models/dcn\_ranker\_metadata.json`



The recorded internal validation AUC is approximately 0.714. The recorded validation loss and training loss suggest that additional tuning and overfitting analysis are warranted.



\## 5. API Layer



The FastAPI service loads the trained ranking model and FAISS index.



Primary endpoints:



| Method | Endpoint               | Purpose                               |

| ------ | ---------------------- | ------------------------------------- |

| GET    | `/`                    | API root                              |

| GET    | `/health`              | Check service and dependency health   |

| GET    | `/recommend/{user\_id}` | Generate personalized recommendations |

| GET    | `/movie/{movie\_id}`    | Retrieve movie metadata               |



The recommendation endpoint accepts a result limit and candidate count. The documented limits are 1–50 recommendations and 10–500 candidates.



The health endpoint reports model availability, index information, and Redis connectivity.



\## 6. Caching



Redis is used to cache recommendation responses.



The configured cache time-to-live is 300 seconds. Response headers can indicate whether a request was served from the cache.



Caching can reduce repeated computation for identical requests. Cache invalidation and behavior under concurrent load require further production testing.



\## 7. Frontend



The frontend is built with React and Vite.



During local development, Vite proxies `/api` requests to the FastAPI service running on port 8002. The proxy removes the `/api` prefix before forwarding the request.



The frontend typically runs at `http://127.0.0.1:5173`.



\## 8. Local Development Ports



| Component         | Local address                |

| ----------------- | ---------------------------- |

| FastAPI           | `http://127.0.0.1:8002`      |

| API documentation | `http://127.0.0.1:8002/docs` |

| React/Vite        | `http://127.0.0.1:5173`      |

| Redis             | `127.0.0.1:6379`             |



These addresses describe the current local development configuration, not a public deployment.



\## 9. End-to-End Request Flow



1\. A user opens the React frontend.

2\. The frontend requests recommendations for a user ID.

3\. FastAPI checks Redis for a cached response.

4\. If a suitable cached response exists, the API returns it.

5\. Otherwise, the retrieval component finds candidate movies using FAISS.

6\. The ranking model scores and reorders the candidates.

7\. The API returns the recommendation results and caches the response.

8\. The frontend displays the returned results.



Exact request handling should be verified against the current implementation in `src/api.py`.



\## 10. Deployment Considerations



Docker configuration is present in the repository. A complete deployment should be considered verified only after all required services start successfully and pass health checks.



Further work should include automated integration tests, input validation, dependency pinning, performance testing, logging, monitoring, and security review.



\## 11. Limitations



This architecture is an educational prototype. It has not been validated as a production recommendation service.



Retrieval quality limits the ranking model because relevant movies absent from the candidate set cannot be recovered by reranking. Offline evaluation results also do not establish real-world click-through-rate or business improvements.
