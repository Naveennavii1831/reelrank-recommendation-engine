from datetime import datetime, timedelta
from pathlib import Path
import json
import math
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
PROJECT = Path(
    os.getenv("REELRANK_PROJECT_DIR", "/opt/reelrank")
)
def check_files(files, category):
    for filename in files:
        path = PROJECT / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"{category} artifact missing: {path}"
            )
        print(f"VERIFIED: {path}")
    print(f"{category} validation successful.")
def validate_datasets():
    check_files([
        "data/processed/train.csv",
        "data/processed/validation.csv",
        "data/processed/test.csv",
        "data/processed/movies_clean.csv",
        "data/processed/movie_features.csv",
        "data/processed/user_features.csv",
        "data/processed/retrieval_train_sample.csv",
    ], "Dataset")
def validate_models():
    check_files([
        "models/dcn_ranker.keras",
        "models/dcn_ranker_compat.h5",
        "models/dcn_ranker_metadata.json",
        "models/two_tower.weights.h5",
    ], "Model")
def validate_faiss():
    check_files([
        "models/movie_faiss.index",
        "models/faiss_movie_ids.npy",
        "models/retrieval_embeddings.npz",
    ], "FAISS")
def validate_documentation():
    check_files([
        "docs/evaluation.md",
        "docs/architecture.md",
        "docs/limitations.md",
        "docs/api.md",
    ], "Documentation")
def load_json(relative_path):
    path = PROJECT / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Report not found: {path}"
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
def validate_ranking_evaluation():
    report = load_json(
        "reports/ranking_evaluation.json"
    )
    results = report.get("results", {})
    required_metrics = [
        "Recall@10",
        "Recall@20",
        "NDCG@10",
        "NDCG@20",
    ]
    for model in ["retrieval", "dcn_reranked"]:
        if model not in results:
            raise ValueError(
                f"Missing ranking results: {model}"
            )
        for metric in required_metrics:
            value = results[model].get(metric)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Invalid {metric} for {model}: {value}"
                )
    retrieval_ndcg = results["retrieval"]["NDCG@10"]
    dcn_ndcg = results["dcn_reranked"]["NDCG@10"]
    print("Ranking evaluation verified.")
    print(json.dumps(results, indent=2))
    if dcn_ndcg <= retrieval_ndcg:
        raise ValueError(
            "DCN NDCG@10 does not exceed retrieval baseline."
        )
    print("DCN NDCG@10 improvement verified.")
def validate_ab_test():
    report = load_json(
        "reports/ab_test_results.json"
    )
    if report.get("experiment_type") != (
        "offline paired ranking comparison"
    ):
        raise ValueError(
            "Unexpected A/B experiment type."
        )
    comparison = report.get("comparison", {})
    for metric in [
        "Recall@10",
        "Recall@20",
        "NDCG@10",
        "NDCG@20",
    ]:
        result = comparison.get(metric)
        if not isinstance(result, dict):
            raise ValueError(
                f"Missing A/B comparison: {metric}"
            )
        improvement = result.get(
            "absolute_difference"
        )
        if (
            not isinstance(improvement, (int, float))
            or not math.isfinite(improvement)
        ):
            raise ValueError(
                f"Invalid improvement for {metric}"
            )
        print(
            f"{metric}: "
            f"{result['retrieval_baseline']:.6f} -> "
            f"{result['dcn_treatment']:.6f}; "
            f"improvement={improvement:.6f}"
        )
    ci = report.get(
        "ndcg_at_10_paired_bootstrap_95_percent_ci",
        {}
    )
    lower = ci.get("ci_95_lower")
    upper = ci.get("ci_95_upper")
    if (
        not isinstance(lower, (int, float))
        or not isinstance(upper, (int, float))
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower > upper
    ):
        raise ValueError(
            "Invalid NDCG@10 bootstrap confidence interval."
        )
    print(
        f"NDCG@10 bootstrap 95% CI: "
        f"[{lower:.6f}, {upper:.6f}]"
    )
    print(
        "Offline A/B comparison verified. "
        "This is not a live online experiment."
    )
with DAG(
    dag_id="reelrank_recommendation_pipeline",
    description=(
        "Validates ReelRank datasets, models, FAISS, "
        "evaluation reports and documentation."
    ),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "reelrank",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=[
        "reelrank",
        "machine-learning",
        "validation",
        "evaluation",
    ],
) as dag:
    datasets = PythonOperator(
        task_id="validate_datasets",
        python_callable=validate_datasets,
    )
    models = PythonOperator(
        task_id="validate_models",
        python_callable=validate_models,
    )
    faiss = PythonOperator(
        task_id="validate_faiss",
        python_callable=validate_faiss,
    )
    ranking = PythonOperator(
        task_id="validate_ranking_evaluation",
        python_callable=validate_ranking_evaluation,
    )
    ab_test = PythonOperator(
        task_id="validate_offline_ab_test",
        python_callable=validate_ab_test,
    )
    documentation = PythonOperator(
        task_id="validate_documentation",
        python_callable=validate_documentation,
    )
    datasets >> models >> faiss
    faiss >> ranking >> ab_test
    ab_test >> documentation
