import json

from experiments.recommendation.compare_recommend import save_results


def test_save_results_writes_json_and_csv(tmp_path) -> None:
    results = [
        {"model_name": "random_unrated", "precision_at_k_mean": 0.01},
        {"model_name": "user_based_cf", "precision_at_k": 0.10},
    ]
    json_path = tmp_path / "metrics.json"
    csv_path = tmp_path / "metrics.csv"

    save_results(results, json_path, csv_path)

    assert json.loads(json_path.read_text()) == results
    assert csv_path.read_text().splitlines()[0] == (
        "model_name,precision_at_k,precision_at_k_mean"
    )
