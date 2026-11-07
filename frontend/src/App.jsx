import { useState } from "react";
import "./App.css";

// Nginx will forward /api requests to the FastAPI backend.
const API_URL = import.meta.env.VITE_API_URL || "/api";

function App() {
  const [userId, setUserId] = useState("1");
  const [limit, setLimit] = useState("10");
  const [recommendations, setRecommendations] = useState([]);
  const [recommendationType, setRecommendationType] = useState("");
  const [candidateCount, setCandidateCount] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchedUser, setSearchedUser] = useState(null);

  async function getRecommendations(event) {
    event.preventDefault();

    const parsedUserId = Number(userId);
    const parsedLimit = Number(limit);

    if (!Number.isInteger(parsedUserId) || parsedUserId <= 0) {
      setError("Please enter a valid positive integer user ID.");
      return;
    }

    if (
      !Number.isInteger(parsedLimit) ||
      parsedLimit < 1 ||
      parsedLimit > 50
    ) {
      setError("Please select between 1 and 50 movies.");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations([]);
    setRecommendationType("");
    setCandidateCount(null);
    setSearchedUser(parsedUserId);

    try {
      const response = await fetch(
  `${API_URL}/recommendations/${parsedUserId}?count=${parsedLimit}&candidate_count=100`
);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to fetch recommendations."
        );
      }

      setRecommendations(data.recommendations || []);
      setRecommendationType(data.recommendation_method || "");
      setCandidateCount(data.candidate_count ?? null);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the recommendation service. Check that the frontend and backend are running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="ReelRank home">
          <span className="brand-icon">R</span>
          <span>ReelRank</span>
        </a>

        <span className="status-pill">
          <span className="status-dot" />
          Recommendation API
        </span>
      </header>

      <section className="hero" id="home">
        <div className="hero-copy">
          <p className="eyebrow">AI-POWERED DISCOVERY</p>

          <h1>
            Your next
            <br />
            favourite film<span>.</span>
          </h1>

          <p className="hero-description">
            Discover movies selected by a two-stage recommendation engine
            powered by candidate retrieval and deep neural ranking.
          </p>
        </div>

        <div className="hero-art" aria-hidden="true">
          <div className="art-circle circle-one" />
          <div className="art-circle circle-two" />

          <div className="art-card">
            <span className="art-star">✦</span>
            <span className="art-caption">
              YOUR NEXT
              <br />
              GREAT WATCH
            </span>
            <span className="art-number">01 — 10</span>
          </div>
        </div>
      </section>

      <section className="search-panel">
        <div>
          <p className="section-label">PERSONALIZED FOR YOU</p>
          <h2>Find your recommendations</h2>

          <p className="muted">
            Enter a MovieLens user ID to generate a ranked movie list.
          </p>
        </div>

        <form className="search-form" onSubmit={getRecommendations}>
          <label>
            MovieLens user ID

            <input
              type="number"
              min="1"
              step="1"
              value={userId}
              onChange={(event) => setUserId(event.target.value)}
              required
            />
          </label>

          <label>
            Number of movies

            <select
              value={limit}
              onChange={(event) => setLimit(event.target.value)}
            >
              <option value="5">5 movies</option>
              <option value="10">10 movies</option>
              <option value="20">20 movies</option>
              <option value="50">50 movies</option>
            </select>
          </label>

          <button
            className="primary-button"
            type="submit"
            disabled={loading}
          >
            {loading ? "Finding movies..." : "Find movies →"}
          </button>
        </form>
      </section>

      {error && (
        <div className="message error-message" role="alert">
          <strong>Something went wrong</strong>
          <p>{error}</p>
          <p>
            If you're running the Docker deployment, check that the frontend
            proxy, FastAPI backend, and Redis service are running.
          </p>
        </div>
      )}

      {loading && (
        <div className="loading-panel" role="status">
          <div className="spinner" />
          <p>Retrieving candidates and ranking movies...</p>
        </div>
      )}

      {!loading && recommendations.length > 0 && (
        <section className="results-section">
          <div className="results-heading">
            <div>
              <p className="section-label">YOUR RESULTS</p>
              <h2>Movies worth discovering</h2>

              <p className="muted">
                User {searchedUser} · {recommendations.length} recommendations
                {candidateCount !== null && ` · ${candidateCount} candidates`}
              </p>
            </div>

            <span className="result-count">
              {recommendations.length} FILMS
            </span>
          </div>

          <div className="results-grid">
            {recommendations.map((movie, index) => (
              <article className="movie-card" key={movie.movie_id}>
                <div className={`movie-art art-${index % 6}`}>
                  <span className="movie-rank">
                    {String(index + 1).padStart(2, "0")}
                  </span>

                  <span className="movie-art-symbol">
                    {["✦", "◈", "✧", "◉", "◇", "✳"][index % 6]}
                  </span>

                  <span className="movie-art-footer">
                    REELRANK / FILM
                  </span>
                </div>

                <div className="movie-info">
                  <div className="movie-title-row">
                    <h3>{movie.title}</h3>
                  </div>

                  <p className="genres">
                    {(movie.genres || "Genre unavailable")
                      .split("|")
                      .join(" · ")}
                  </p>

                  {typeof movie.score === "number" && (
                    <div className="score-block">
                      <div className="score-heading">
                        <span>Ranking score</span>
                        <strong>{movie.score.toFixed(3)}</strong>
                      </div>

                      <div className="score-track">
                        <div
                          className="score-fill"
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(100, movie.score * 100)
                            )}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}

                  {movie.retrieval_similarity !== null &&
                    movie.retrieval_similarity !== undefined && (
                      <p className="similarity">
                        Retrieval similarity:{" "}
                        {movie.retrieval_similarity.toFixed(3)}
                      </p>
                    )}

                  <span className="recommendation-tag">
                    {movie.recommendation_type === "popularity_fallback"
                      ? "Popular pick"
                      : "AI ranked"}
                  </span>
                </div>
              </article>
            ))}
          </div>

          <p className="disclaimer">
            Experimental recommendations. Ranking scores are model outputs,
            not calibrated probabilities. Results may be inaccurate.
          </p>
        </section>
      )}

      {!loading && !error && recommendations.length === 0 && (
        <section className="empty-state">
          <div className="empty-symbol">✳</div>

          <h2>Your next movie is waiting</h2>

          <p>
            Enter a user ID above and select "Find movies" to see your
            recommendations.
          </p>
        </section>
      )}

      <footer className="footer">
        <span>REELRANK · AI RECOMMENDATION PROTOTYPE</span>
        <span>Two-tower retrieval + FAISS + DCN ranking</span>
      </footer>
    </main>
  );
}

export default App;
