import json


def _run_check():
    with open("dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/autometrics/results/results.json") as f:
        metrics_data = json.load(f)

    current_date = list(metrics_data.keys())[-1]
    current_metrics = metrics_data.get(current_date, {})
    previous_date = list(metrics_data.keys())[-2]
    previous_metrics = metrics_data.get(previous_date, {})

    differences = {}
    for algorithm, metrics in current_metrics.items():
        if algorithm in previous_metrics:
            differences[algorithm] = {
                "all_paths_sampled_diff": metrics.get("all_paths_sampled_avg", 0) - previous_metrics[algorithm].get("all_paths_sampled_avg", 0),
                "all_utterances_present_diff": metrics.get("all_utterances_present_avg", 0)
                - previous_metrics[algorithm].get("all_utterances_present_avg", 0),
                "all_roles_correct_diff": metrics.get("all_roles_correct_avg", 0) - previous_metrics[algorithm].get("all_roles_correct_avg", 0),
                "is_correct_length_diff": metrics.get("is_correct_lenght_avg", 0) - previous_metrics[algorithm].get("is_correct_lenght_avg", 0),
                "are_triplets_valid_diff": metrics.get("are_triplets_valid", 0) - previous_metrics[algorithm].get("are_triplets_valid", 0),
                "is_theme_valid_diff": metrics.get("is_theme_valid_avg", 0) - previous_metrics[algorithm].get("is_theme_valid_avg", 0),
                "total_diff": (
                    metrics.get("all_paths_sampled_avg", 0)
                    + metrics.get("all_utterances_present_avg", 0)
                    + metrics.get("all_roles_correct_avg", 0)
                    + metrics.get("is_correct_lenght_avg", 0)
                    + metrics.get("are_triplets_valid", 0)
                    + metrics.get("is_theme_valid_avg", 0)
                )
                - (
                    previous_metrics[algorithm].get("all_paths_sampled_avg", 0)
                    + previous_metrics[algorithm].get("all_utterances_present_avg", 0)
                    + previous_metrics[algorithm].get("all_roles_correct_avg", 0)
                    + previous_metrics[algorithm].get("is_correct_lenght_avg", 0)
                    + previous_metrics[algorithm].get("are_triplets_valid", 0)
                    + metrics.get("is_theme_valid_avg", 0)
                ),
            }

    for algorithm, diff in differences.items():
        print(f"Algorithm: {algorithm}")
        print(f"Differences: {diff}")
        if diff["total_diff"] < 0:
            raise ValueError(f"Total difference for {algorithm} is negative: {diff['total_diff']}")

    print("Everything is fine.")
