import logging
import argparse
import json
import math
from collections import Counter, defaultdict

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


def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k.
    n: total number of samples
    c: number of correct samples
    k: k in pass@k
    """
    if n < k:
        return 1.0 if c > 0 else 0.0

    if c == n:
        return 1.0

    # 1 - binom(n-c, k) / binom(n, k)
    try:
        prob_fail = math.comb(n - c, k) / math.comb(n, k)
        return 1.0 - prob_fail
    except Exception:
        return 0.0


def _evaluate_jsonl_file(file_path: str, sub_category: str = None) -> dict:

    # Storage for grouping by problem ID
    # Key: (id, sentence_idx) -> List of {predicted, is_correct}
    problems = defaultdict(list)

    total_tokens = 0
    total_entries = 0

    parent_category = get_parent_category(sub_category) if sub_category else None

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            for line in f:
                item = json.loads(line.strip())
                total_entries += 1
                total_tokens += item.get("token_length", 0)

                is_correct = False
                predicted = None

                if parent_category == "general":
                    ground_truth = item["answer"]
                    predicted = extract_multiple_choice_answer(item["response"])
                    is_correct = verify_multiple_choice(ground_truth, predicted)
                elif parent_category == "math":
                    ground_truth = parse(f"${item['answer']}$")
                    parsed_pred = parse(item["response"])
                    predicted = str(parsed_pred)  # Store as string for voting
                    is_correct = verify(ground_truth, parsed_pred)
                elif parent_category == "tool":
                    if sub_category in ["bfcl-v1", "bfcl-v2"]:
                        pred_obj, ground_truth, is_correct = evaluate_bfcl_entry(item)
                        predicted = str(pred_obj)  # serialized for voting
                    else:
                        ground_truth = None
                        predicted = None
                        pass
                else:
                    raise ValueError(
                        f"Unknown parent category found: {parent_category}"
                    )

                # Identify unique problem
                problem_id = item.get("id", "unknown")
                sentence_idx = item.get("sentence_idx", -1)
                key = (problem_id, sentence_idx)

                problems[key].append(
                    {
                        "predicted": predicted,
                        "is_correct": is_correct,
                        "ground_truth": ground_truth,
                    }
                )

        except Exception as e:
            logging.error(f"Error evaluating the file {file_path}: {e}")
            return None

    # Aggregation
    num_problems = len(problems)
    if num_problems == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "avg_token_length": 0.0,
            "pass@k": {},
            "majority_accuracy": 0.0,
        }

    sum_accuracy = 0.0
    sum_majority = 0.0

    # Initialize pass@k sums. We'll track k=1, 2, 4, 8, etc. based on max trials
    max_trials = 0
    for trials in problems.values():
        max_trials = max(max_trials, len(trials))

    # Define Ks to track
    ks_to_track = [1, 2, 4, 5, 8, 10, 16, 32, 64, 100]
    ks_to_track = [k for k in ks_to_track if k <= max_trials]
    if max_trials not in ks_to_track and max_trials > 0:
        ks_to_track.append(max_trials)
    ks_to_track.sort()

    pass_at_k_sums = defaultdict(float)

    for key, trials in problems.items():
        n = len(trials)
        c = sum(1 for t in trials if t["is_correct"])

        # 1. Average Accuracy (Pass@1 effectively, averaged over problems)
        sum_accuracy += c / n

        # 2. Majority Vote
        # Gather predictions
        if parent_category == "general":
            # For multiple choice, predicted is just "A", "B", etc.
            preds = [t["predicted"] for t in trials if t["predicted"]]
        else:
            preds = [t["predicted"] for t in trials]

        if preds:
            # Find most common
            counter = Counter(preds)
            most_common = counter.most_common(1)
            majority_pred = most_common[0][0]

            # Verify majority prediction
            # We need to re-verify because 'is_correct' is stored, but we need to know if the majority consensus is correct.
            # However, for simple cases, we can check if the majority_pred matches any correct prediction's string.
            # But wait, 'is_correct' was determined by verifying(ground_truth, predicted).
            # If majority_pred is one of the predictions that was marked correct, then it is correct.
            # Assuming verifying is deterministic for the same string.

            # Check if majority_pred corresponds to a correct trial
            majority_is_correct = False
            for t in trials:
                if t["predicted"] == majority_pred and t["is_correct"]:
                    majority_is_correct = True
                    break
                # Special handling if verification is complex (like float comparison)
                # But we stored 'predicted' as string/parsed object.
                # If parsed objects are different but evaluate to same correctness, grouping by string might be imperfect
                # but it's the standard way for majority voting (exact match consistency).

            if majority_is_correct:
                sum_majority += 1.0

        # 3. Pass@k
        for k in ks_to_track:
            pass_at_k_sums[k] += calculate_pass_at_k(n, c, k)

    metrics = {
        "total": total_entries,
        "num_problems": num_problems,
        "avg_token_length": (
            (total_tokens / total_entries) if total_entries > 0 else 0.0
        ),
        "accuracy": (sum_accuracy / num_problems * 100),
        "majority_accuracy": (sum_majority / num_problems * 100),
        "pass@k": {k: (pass_at_k_sums[k] / num_problems * 100) for k in ks_to_track},
    }

    return metrics


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
                    "accuracy": eval_result["accuracy"],
                    "majority_accuracy": eval_result["majority_accuracy"],
                    "avg_token_length": eval_result["avg_token_length"],
                    "pass@k": eval_result["pass@k"],
                }
                logging.info("")
                logging.info(f"Category: {sub_category} / Threshold: {threshold_value}")
                logging.info(f"- Avg Acc (Pass@1) : {eval_result['accuracy']:.2f}%")
                logging.info(
                    f"- Majority Vote    : {eval_result['majority_accuracy']:.2f}%"
                )

                # Log Pass@K
                pk_str = ", ".join(
                    [f"k={k}: {v:.1f}%" for k, v in eval_result["pass@k"].items()]
                )
                logging.info(f"- Pass@K           : {pk_str}")

                logging.info(
                    f"- Avg Tokens       : {eval_result['avg_token_length']:.0f}"
                )
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
