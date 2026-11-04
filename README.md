# ReelRank – Large-Scale Personalized Recommendation Engine

A two-stage personalized movie recommendation system built using deep learning, vector similarity search, neural ranking, and scalable inference infrastructure.

ReelRank uses a Two-Tower retrieval architecture to identify candidate movies and a Deep & Cross Network (DCN) to rerank them based on predicted relevance.

The project uses the MovieLens 25M dataset and integrates FAISS, TensorFlow, NVIDIA Triton Inference Server, FastAPI, Redis, React, Docker, and Apache Airflow.

---

## 1. Project Overview

Modern recommendation systems must retrieve relevant items from large catalogs and rank them efficiently.

ReelRank implements a two-stage recommendation architecture:

1. **Candidate Retrieval:** A Two-Tower neural network generates user and movie embeddings. FAISS performs efficient similarity search to retrieve candidate movies.
2. **Candidate Ranking:** A DCN-based ranking model scores and reranks the retrieved candidates.

The system also includes popularity-based fallback recommendations for unknown users.

### Key Objectives

- Build a personalized recommendation engine.
- Implement scalable candidate retrieval.
- Improve ranking quality using deep learning.
- Support efficient vector similarity search.
- Deploy model inference through Triton.
- Reduce repeated computation using Redis caching.
- Provide a frontend for interacting with recommendations.
- Automate artifact validation using Apache Airflow.

---

## 2. Key Features

- MovieLens 25M dataset preprocessing.
- Chronological training, validation, and test splitting.
- Two-Tower neural retrieval model.
- 32-dimensional user and movie embeddings.
- FAISS vector similarity search.
- Deep & Cross Network (DCN) ranking model.
- ONNX model conversion.
- NVIDIA Triton inference server.
- FastAPI recommendation backend.
- Redis caching with a 300-second TTL.
- Popularity-based cold-start fallback.
- React frontend.
- Offline Recall@K and NDCG@K evaluation.
- Offline paired ranking comparison with bootstrap confidence intervals.
- Apache Airflow artifact-validation pipeline.
- Docker-based service configuration.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Dataset | MovieLens 25M |
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Deep Learning | TensorFlow |
| Retrieval Architecture | Two-Tower |
| Vector Search | FAISS |
| Ranking Architecture | DCN |
| Model Format | ONNX |
| Model Serving | NVIDIA Triton |
| Backend | FastAPI, Uvicorn |
| Caching | Redis |
| Frontend | React, Vite |
| Workflow Orchestration | Apache Airflow |
| Containerization | Docker |
| Evaluation | Recall@K, NDCG@K |

---

## 4. System Architecture

```text
                 MovieLens 25M Dataset
                          |
                          v
                 Data Preprocessing
                          |
                          v
                 Chronological Split
                          |
                 +--------+--------+
                 |                 |
                 v                 v
          Two-Tower Model    Feature Engineering
                 |
                 v
          User/Movie Embeddings
                 |
                 v
              FAISS
          Candidate Retrieval
                 |
                 v
          Candidate Movie List
                 |
                 v
           DCN Ranking Model
                 |
                 v
          ONNX Model Inference
                 |
                 v
        NVIDIA Triton Server
                 |
                 v
           FastAPI Backend
                 |
           +-----+------+
           |            |
           v            v
       Redis Cache   Fallback Logic
           |            |
           +-----+------+
                 |
                 v
            React UI
                 |
                 v
         Movie Recommendations
```

### Recommendation Workflow

1. User submits a user ID and requested recommendation count.
2. The backend retrieves user embeddings.
3. FAISS searches for candidate movies.
4. The DCN ranking model scores the candidates through Triton.
5. Candidates are sorted by predicted relevance.
6. Redis caches the response.
7. If personalized retrieval returns no candidates, the system uses popularity-based fallback recommendations.
8. The frontend displays the results.

---

## 5. Project Structure

```text
recommendation-engine/
│
├── airflow/
│   ├── dags/
│   │   └── recommendation_pipeline.py
│   ├── logs/
│   └── requirements.txt
│
├── data/
│   └── processed/
│
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── evaluation.md
│   └── limitations.md
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── dcn_ranker.keras
│   ├── dcn_ranker_compat.h5
│   ├── dcn_ranker_metadata.json
│   ├── two_tower.weights.h5
│   ├── movie_faiss.index
│   ├── faiss_movie_ids.npy
│   └── retrieval_embeddings.npz
│
├── reports/
│   ├── ranking_evaluation.json
│   ├── ab_test_results.json
│   ├── retrieval_evaluation.json
│   └── training reports
│
├── src/
│   ├── api.py
│   ├── prepare_data.py
│   ├── train_retrieval.py
│   ├── build_faiss_index.py
│   ├── train_ranking.py
│   ├── evaluate_ranking.py
│   ├── evaluate_retrieval.py
│   ├── ab_test_rankers.py
│   └── convert_dcn_to_onnx_v2.py
│
├── tests/
│
├── triton/
│   └── models/
│       └── dcn_ranker/
│           ├── config.pbtxt
│           └── 1/
│               └── model.onnx
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 6. Model Evaluation

The ranking evaluation uses chronological validation data and a relevance threshold of rating ≥ 4.0.

### Retrieval vs DCN Ranking

| Metric | FAISS Retrieval | DCN Reranking |
|---|---:|---:|
| Recall@10 | 0.002489 | 0.007033 |
| Recall@20 | 0.004699 | 0.010783 |
| NDCG@10 | 0.006752 | 0.012125 |
| NDCG@20 | 0.007448 | 0.012816 |

### Relative Improvement

| Metric | Improvement |
|---|---:|
| Recall@10 | 182.5% |
| Recall@20 | 129.5% |
| NDCG@10 | 79.6% |
| NDCG@20 | 72.1% |

### Experimental Details

- Evaluation users selected: 500.
- Users with retrieved candidates: 475.
- Users with relevant items among candidates: 151.
- Candidate movies per user: 100.
- Bootstrap samples: 2,000.
- NDCG@10 mean paired difference: approximately 0.00537.
- NDCG@10 paired bootstrap 95% confidence interval: approximately 0.00110–0.00998.

The experiment indicates improved ranking performance after DCN reranking on the evaluated cohort.

These are offline evaluation results, not results from a live randomized A/B test.

Detailed results:

- `reports/ranking_evaluation.json`
- `reports/ab_test_results.json`
- `reports/retrieval_evaluation.json`

---

## 7. Running the Backend

Open PowerShell in the project root.

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Set TensorFlow compatibility:

```powershell
$env:TF_USE_LEGACY_KERAS = "1"
```

Start FastAPI:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8001
```

Backend:

http://127.0.0.1:8001

API documentation:

http://127.0.0.1:8001/docs

Health endpoint:

http://127.0.0.1:8001/health

The backend requires the model artifacts, FAISS index, Triton inference service, and Redis service to be available.

---

## 8. Running the Frontend

Open another PowerShell terminal:

```powershell
cd D:\AI-ML-Projects\recommendation-engine\frontend
```

Install dependencies if necessary:

```powershell
npm ci
```

Start the development server:

```powershell
npm run dev
```

Open:

http://localhost:5173

---

## 9. API Usage

### Health Check

```http
GET /health
```

### Get Recommendations

```http
GET /recommendations/{user_id}?count=10&candidate_count=100
```

Example:

```http
GET http://127.0.0.1:8001/recommendations/100?count=10&candidate_count=100
```

### Unknown User / Cold-Start Example

```http
GET http://127.0.0.1:8001/recommendations/9999?count=50&candidate_count=500
```

When personalized retrieval returns no candidates, the system switches to popularity-based fallback recommendations.

Example response fields:

```json
{
  "count": 50,
  "recommendation_method": "popularity_fallback",
  "cache_status": "MISS",
  "message": "Personalized retrieval returned no candidates. Popularity fallback was used."
}
```

---

## 10. Triton Model Serving

The DCN ranking model is converted to ONNX and served using NVIDIA Triton Inference Server.

Configured endpoints:

| Service | Port |
|---|---:|
| HTTP | 8003 |
| gRPC | 8004 |
| Metrics | 8005 |

Model metadata:

```http
GET http://localhost:8003/v2/models/dcn_ranker
```

Triton model inference is configured for CPU execution.

---

## 11. Redis Caching

Redis is used to cache recommendation responses.

Configuration:

- Redis version: 7 Alpine.
- Port: 6379.
- Cache TTL: 300 seconds.

The API response exposes cache status:

- `MISS`: Recommendation response generated and cached.
- `HIT`: Cached recommendation response returned.

---

## 12. Apache Airflow Orchestration

The project includes an Airflow DAG:

`reelrank_recommendation_pipeline`

The manually triggered workflow validates:

1. Dataset availability.
2. Model artifacts.
3. FAISS index files.
4. Ranking evaluation results.
5. Offline paired comparison results.
6. Project documentation.

The DAG completed successfully in the verified execution.

The current workflow validates existing artifacts and reports. It does not retrain models or overwrite evaluation results.

Airflow dashboard:

http://localhost:8080

---

## 13. Limitations

- This is an educational machine-learning prototype.
- Offline evaluation metrics are relatively low and require further improvement.
- The ranking model can only rerank movies retrieved by FAISS.
- Model coverage is limited to known users and movies.
- Training uses sampled historical interactions.
- The offline comparison is not a live online A/B test.
- CTR cannot be measured without impression and click logs.
- Popularity fallback does not provide personalized recommendations.
- Production load testing, security, monitoring, and reliability validation remain necessary.
- Docker Compose configuration is available, but the complete multi-service deployment requires separate end-to-end verification.

---

## 14. Future Improvements

- Improve retrieval candidate coverage.
- Experiment with embedding dimensions and model architectures.
- Improve ranking model performance.
- Add content-based recommendations for new users and movies.
- Introduce real-time interaction features.
- Implement impression and click tracking.
- Conduct live A/B experiments.
- Add automated integration and load tests.
- Improve production monitoring and deployment.

---

## 15. Disclaimer

ReelRank is an educational machine-learning prototype developed for demonstrating large-scale recommendation system concepts.

The reported metrics apply only to the documented dataset splits, candidate-generation process, and evaluation cohorts. They should not be interpreted as guaranteed production performance.