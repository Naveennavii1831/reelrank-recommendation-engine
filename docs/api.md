\# ReelRank API Documentation



\## 1. Overview



ReelRank exposes its movie recommendation functionality through a FastAPI service.



\*\*Local base URL:\*\* `http://127.0.0.1:8002`



\*\*Interactive API documentation:\*\* `http://127.0.0.1:8002/docs`



\## 2. Health Check



\*\*Endpoint:\*\* `GET /health`



Checks the availability of the recommendation service and its principal dependencies.



The previously verified response reported:



\* Service status: healthy

\* Ranking model loaded: true

\* FAISS index loaded: true

\* Indexed movies: 10,027

\* User embeddings: 85,244

\* Redis connected: true

\* Cache TTL: 300 seconds



These values describe the recorded local environment and may change.



\## 3. Generate Recommendations



\*\*Endpoint:\*\* `GET /recommend/{user\_id}`



Returns personalized movie recommendations for a user.



\### Parameters



| Parameter    | Type                    |  Default | Documented range               |

| ------------ | ----------------------- | -------: | ------------------------------ |

| `user\_id`    | Integer/path parameter  | Required | Must identify a supported user |

| `limit`      | Integer/query parameter |       10 | 1–50                           |

| `candidates` | Integer/query parameter |      100 | 10–500                         |



\### Example request



`GET http://127.0.0.1:8002/recommend/2?limit=10\&candidates=100`



The service retrieves candidate movies and uses the ranking model to order them.



The exact JSON response structure should be checked in the running API's interactive documentation.



\### Important interpretation



Ranking scores are model scores, not calibrated probabilities. A score should not be interpreted as the probability that a user will watch or enjoy a movie unless calibration has been separately evaluated.



\## 4. Retrieve Movie Metadata



\*\*Endpoint:\*\* `GET /movie/{movie\_id}`



Retrieves metadata for a movie identified by its movie ID.



Use the interactive API documentation to confirm the exact response fields and error behavior.



\## 5. Cache Behavior



Redis caches recommendation responses using a configured TTL of 300 seconds.



The local implementation was tested with repeated requests that returned cache MISS and HIT indicators in the `X-Cache` response header.



A cache HIT indicates that a cached response was served. A cache MISS indicates that the request required the uncached recommendation path.



\## 6. Run the API Locally



From the project root, activate the existing virtual environment and configure TensorFlow compatibility:



`$env:TF\_USE\_LEGACY\_KERAS = "1"`



Start the API:



`python -m uvicorn src.api:app --host 127.0.0.1 --port 8002`



Open:



\* Health: `http://127.0.0.1:8002/health`

\* API explorer: `http://127.0.0.1:8002/docs`



Keep the backend terminal running while testing.



\## 7. Error Handling and Validation



Input validation and error responses should be verified using the running API documentation.



Recommended additional tests include:



\* Invalid user IDs.

\* Unsupported movie IDs.

\* Out-of-range recommendation limits.

\* Out-of-range candidate counts.

\* Missing model artifacts.

\* Redis unavailable.

\* Empty candidate results.

\* Repeated requests and cache expiration.



Do not assume an error response is implemented until it has been tested.



\## 8. Security Considerations



This API is intended for local educational use. Before public deployment, add appropriate authentication, request limits, stricter input validation, logging, monitoring, and security controls.
