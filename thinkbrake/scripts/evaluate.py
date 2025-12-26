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
    evaluate_bfcl_entry,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s [ThinkBrake] %(message)s ",
    datefmt="%m-%d %H:%M:%S",
)


def _evaluate_jsonl_file(file_path: str, sub_category: str = None) -> dict:
    results = []
    correct_count = 0
    total_count = 0
    total_tokens = 0

    parent_category = get_parent_category(sub_category) if sub_category else None

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            for line in f:
                item = json.loads(line.strip())
                is_correct = False

                if parent_category == "general":
                    ground_truth = item["answer"]
                    predicted = extract_multiple_choice_answer(item["response"])
                    is_correct = verify_multiple_choice(ground_truth, predicted)
                elif parent_category == "math":
                    ground_truth = parse(f"${item['answer']}$")
                    predicted = parse(item["response"])
                    is_correct = verify(ground_truth, predicted)
                elif parent_category == "tool":
                    if sub_category in ["bfcl-v1", "bfcl-v2"]:
                        predicted, ground_truth, is_correct = evaluate_bfcl_entry(item)
                    else:
                        ground_truth = None
                        predicted = None
                        pass
                else:
                    raise ValueError(
                        f"Unknown parent category found: {parent_category}"
                    )

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
    model_path = Path(model_path)
    all_results = {}

    if not model_path.exists():
        return None

    for parent_category in model_path.iterdir():
        if not parent_category.is_dir():
            continue

        target_dir = parent_category / THINKBRAKE_PREFIX
        if not target_dir.exists():
            continue

        try:
            threshold_dirs = sorted(
                [d for d in target_dir.iterdir()],
                key=lambda d: float(d.name.replace("threshold_", "")),
            )
        except ValueError:
            threshold_dirs = sorted([d for d in target_dir.iterdir()])

        relevant_categories = [
            cat
            for cat in categories
            if get_parent_category(cat) == parent_category.name
        ]

        for sub_category in relevant_categories:
            for threshold_dir in threshold_dirs:
                threshold_value = threshold_dir.name.replace("threshold_", "")
                jsonl_file = threshold_dir / f"{sub_category}_result.jsonl"

                if not jsonl_file.exists():
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
    results = _evaluate_model_results(model_path, categories)

    if results is None:
        return None

    structured_entry = {}
    for category_key, threshold_map in results.items():
        for threshold_key, stats in threshold_map.items():
            structured_entry.setdefault(category_key, {})
            structured_entry[category_key][threshold_key] = stats

    return structured_entry


def _save_leaderboard_entry(model_name: str, entry: dict):
    leaderboard_file = RESULT_DIR / f"leaderboard_{THINKBRAKE_PREFIX}.json"
    leaderboard = {}
    if leaderboard_file.exists():
        try:
            with open(leaderboard_file, "r", encoding="utf-8") as f:
                leaderboard = json.load(f)
        except Exception:
            pass

    if model_name not in leaderboard:
        leaderboard[model_name] = {}

    leaderboard[model_name].update(entry)

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

        real_model_name = model

        entry = _generate_leaderboard_entry(str(model_path), categories)
        if entry:
            _save_leaderboard_entry(real_model_name, entry)


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
