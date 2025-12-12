import logging
import argparse
import json

from pathlib import Path
from math_verify import parse, verify

from thinkbrake.func.constants import RESULT_DIR, THINKBRAKE_PREFIX
from thinkbrake.func.utils import (
    get_models,
    get_parent_category,
    extract_multiple_choice_answer,
    get_test_categories,
    verify_multiple_choice,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [ThinkBrake] %(message)s ",
    datefmt="%m-%d %H:%M:%S",
)


def _evaluate_jsonl_file(file_path: str, sub_category: str = None) -> dict:
    """
    Evaluates a single JSONL file, calculating accuracy and token usage.
    Supports both math (using math_verify) and general multiple choice tasks.
    """
    results = []
    correct_count = 0
    total_count = 0
    total_tokens = 0

    parent_category = get_parent_category(sub_category) if sub_category else None
    is_multiple_choice = parent_category == "general"

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            for line in f:
                item = json.loads(line.strip())

                if is_multiple_choice:
                    ground_truth = item["answer"]
                    predicted = extract_multiple_choice_answer(item["response"])
                    is_correct = verify_multiple_choice(ground_truth, predicted)
                else:
                    ground_truth = parse(f"${item['answer']}$")
                    predicted = parse(item["response"])
                    is_correct = verify(ground_truth, predicted)

                if is_correct:
                    correct_count += 1

                total_count += 1
                total_tokens += item["token_length"]

                results.append(
                    {
                        "id": item.get("id", f"item_{total_count}"),
                        "ground_truth": ground_truth,
                        "predicted": predicted,
                        "correct": is_correct,
                        "token_length": item["token_length"],
                    }
                )
        except Exception as e:
            logging.error(f"Error evaluating the file {file_path}: {e}")
            return None

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    avg_token_length = (total_tokens / total_count) if total_count > 0 else 0.0

    return {
        "total": total_count,
        "correct": correct_count,
        "accuracy": accuracy,
        "avg_token_length": avg_token_length,
    }


def _evaluate_model_results(model_path: str, categories: list[str]) -> dict:
    """
    Iterates through all result files for a model and evaluates them.
    """
    model_path = Path(model_path)
    all_results = {}

    if not model_path.exists():
        return None

    for parent_category in model_path.iterdir():
        target_dir = parent_category / THINKBRAKE_PREFIX

        for threshold_dir in target_dir.iterdir():
            threshold_value = threshold_dir.name.replace("threshold_", "")
            for jsonl_file in threshold_dir.glob("*_result.jsonl"):
                sub_category = jsonl_file.stem.replace("_result", "")
                if sub_category not in categories:
                    continue

                eval_result = _evaluate_jsonl_file(str(jsonl_file), sub_category)
                if not eval_result:
                    logging.info(
                        f"Skipping the category {sub_category} / threshold: {threshold_value} for unexpected errors."
                    )
                    continue

                all_results.setdefault(sub_category, {})
                all_results[sub_category][threshold_dir.name] = {
                    "total": eval_result["total"],
                    "correct": eval_result["correct"],
                    "accuracy": eval_result["accuracy"],
                    "avg_token_length": eval_result["avg_token_length"],
                }
                logging.info("")
                logging.info(f"Category: {sub_category} / Threshold: {threshold_value}")
                logging.info(
                    f"- Accuracy  : {eval_result['accuracy']:.2f}% ({eval_result['correct']}/{eval_result['total']})"
                )
                logging.info(f"- Avg Tokens: {eval_result['avg_token_length']:.0f}")
                logging.info("")

    return all_results


def _generate_leaderboard_entry(model_path: str, categories: list[str]):
    """
    Generates a summary dictionary for the leaderboard.
    """
    results = _evaluate_model_results(model_path, categories)

    if results is None:
        return None

    leaderboard_entry = {}
    for threshold in results.keys():
        for category_key, stats in results[threshold].items():
            leaderboard_entry.setdefault(threshold, {})
            leaderboard_entry[threshold][category_key] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy"],
                "avg_token_length": stats["avg_token_length"],
            }

    return leaderboard_entry


def _save_leaderboard_entry(model_name: str, entry: dict):
    """
    Saves the leaderboard entry to a JSON file.
    """
    leaderboard_file = RESULT_DIR / f"leaderboard_{THINKBRAKE_PREFIX}.json"
    if leaderboard_file.exists():
        with open(leaderboard_file, "r", encoding="utf-8") as f:
            leaderboard = json.load(f)
            leaderboard.update({model_name: entry})
    else:
        leaderboard = {model_name: entry}

    with open(leaderboard_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)


def main(args):
    models = get_models(args.model)
    categories = get_test_categories(args.category)
    for model in models:
        model_name = model.replace("/", "_")
        model_path = RESULT_DIR / model_name
        if not model_path.exists():
            return

        logging.info(f"Evaluating the model {model}...")
        model_name = model_path.name.replace("_", "/")
        entry = _generate_leaderboard_entry(str(model_path), categories)
        if entry:
            _save_leaderboard_entry(model_name, entry)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        help="Pretrained model name or local path to use for both rollout and sentence generation",
    )
    argparser.add_argument(
        "--category",
        type=str,
        default="aime2024 aime2025",
        help="Test dataset category or multiple categories to evaluate (e.g., math500, gsm8k)",
    )
    return argparser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
