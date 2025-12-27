import logging
import argparse
import json
import math
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import threading
import asyncio
from typing import Optional

from math_verify import parse, verify
from tqdm import tqdm

from thinkbrake.func.constants import RESULT_DIR, ROLLOUT_PREFIX
from thinkbrake.func.handler import BaseHandler
from thinkbrake.func.utils import (
    collect_test_cases,
    get_handler,
    get_models,
    get_parent_category,
    save_result,
    get_test_categories,
    extract_multiple_choice_answer,
    verify_multiple_choice,
    evaluate_bfcl_entry,
    get_test_case_id,
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


async def _process_entry_async(
    handler: BaseHandler,
    entry: dict,
    reasoning_tokens_budget: Optional[int],
    answer_tokens_budget: Optional[int],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        result = await handler.inference_async(
            entry,
            reasoning_tokens_budget=reasoning_tokens_budget,
            answer_tokens_budget=answer_tokens_budget,
            threshold=None,
        )

    return {**entry, **result}


async def _generate_results_async(args, model, entries: list[dict]) -> None:
    loop = asyncio.get_running_loop()
    pool_size = args.num_workers + 4
    loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

    handler: BaseHandler = get_handler(model)
    handler.spin_up_local_server(
        tensor_parallel_size=args.tensor_parallel_size,
        max_total_tokens=args.max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
    )

    num_workers = args.num_workers
    write_queue = queue.Queue()

    def _writer():
        while True:
            item = write_queue.get()
            if item is None:
                break
            save_result(model, item, prefix=ROLLOUT_PREFIX)
            write_queue.task_done()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    try:
        id_to_entry = {get_test_case_id(entry): entry for entry in entries}
        ready_queue = deque([get_test_case_id(entry) for entry in entries])

        in_flight = set()
        sem = asyncio.Semaphore(num_workers)

        with tqdm(
            total=len(entries),
            desc=f"Generating results for {model}",
        ) as pbar:

            while ready_queue or in_flight:
                while ready_queue:
                    entry_id = ready_queue.popleft()
                    entry = id_to_entry[entry_id]

                    task = asyncio.create_task(
                        _process_entry_async(
                            handler,
                            entry,
                            args.reasoning_tokens_budget,
                            args.answer_tokens_budget,
                            sem,
                        )
                    )
                    task.set_name(entry_id)
                    in_flight.add(task)

                if not in_flight:
                    break

                done, in_flight = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    entry_id = task.get_name()
                    try:
                        result = await task
                        write_queue.put(result)
                    except Exception as e:
                        logging.error(f"Error processing entry {entry_id}: {e}")

                    pbar.update()

                    for child_id in children_of[entry_id]:
                        dependencies[child_id].discard(entry_id)
                        if not dependencies[child_id]:
                            ready_queue.append(child_id)

    finally:
        write_queue.put(None)
        writer_thread.join()
        handler.shutdown_local_server()


def _evaluate_jsonl_file(file_path: str, sub_category: str = None) -> dict:
    # Storage for grouping by problem ID
    # Key: (id, sentence_idx) -> List of {predicted, is_correct, ground_truth}
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
                        predicted = str(pred_obj)
                    else:
                        ground_truth = None
                        predicted = None
                        pass
                else:
                    raise ValueError(f"Unknown category found: {parent_category}")

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

    # Initialize pass@k sums
    max_trials = 0
    for trials in problems.values():
        max_trials = max(max_trials, len(trials))

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
        if parent_category == "general":
            preds = [t["predicted"] for t in trials if t["predicted"]]
        else:
            # For math/tool, predicted is already stringified or None
            preds = [t["predicted"] for t in trials if t["predicted"] is not None]

        if preds:
            counter = Counter(preds)
            most_common = counter.most_common(1)
            majority_pred = most_common[0][0]

            # Check if majority_pred corresponds to a correct trial
            majority_is_correct = False
            for t in trials:
                # We compare strings here.
                # Note: t["predicted"] is the string representation.
                if t["predicted"] == majority_pred and t["is_correct"]:
                    majority_is_correct = True
                    break

            if majority_is_correct:
                sum_majority += 1.0

        # 3. Pass@k
        for k in ks_to_track:
            pass_at_k_sums[k] += calculate_pass_at_k(n, c, k)

    return {
        "total": total_entries,
        "correct": sum(
            [sum(1 for t in p if t["is_correct"]) for p in problems.values()]
        ),
        "accuracy": (sum_accuracy / num_problems * 100),
        "majority_accuracy": (sum_majority / num_problems * 100),
        "avg_token_length": (
            (total_tokens / total_entries) if total_entries > 0 else 0.0
        ),
        "pass@k": {k: (pass_at_k_sums[k] / num_problems * 100) for k in ks_to_track},
    }


def _evaluate_model_results(model_path: str, categories: list[str]) -> dict:
    model_path = Path(model_path)
    all_results = {}

    if not model_path.exists():
        return None

    for parent_category in model_path.iterdir():
        target_dir = parent_category / ROLLOUT_PREFIX

        for jsonl_file in target_dir.glob("*_result.jsonl"):
            sub_category = jsonl_file.stem.replace("_result", "")
            # Relax filtering if needed, but stick to requested categories for safety
            if sub_category not in categories:
                continue

            eval_result = _evaluate_jsonl_file(str(jsonl_file), sub_category)

            # Pack results
            all_results[sub_category] = {
                "total": eval_result["total"],
                "accuracy": eval_result["accuracy"],
                "majority_accuracy": eval_result["majority_accuracy"],
                "avg_token_length": eval_result["avg_token_length"],
                "pass@k": eval_result["pass@k"],
            }

            logging.info("")
            logging.info(f"Category: {sub_category}")
            logging.info(f"- Avg Acc (Pass@1) : {eval_result['accuracy']:.2f}%")
            logging.info(
                f"- Majority Vote    : {eval_result['majority_accuracy']:.2f}%"
            )
            pk_str = ", ".join(
                [f"k={k}: {v:.1f}%" for k, v in eval_result["pass@k"].items()]
            )
            logging.info(f"- Pass@K           : {pk_str}")
            logging.info(f"- Avg Tokens       : {eval_result['avg_token_length']:.0f}")
            logging.info("")

    return all_results


def _generate_leaderboard_entry(model_path: str, categories: list[str]):
    results = _evaluate_model_results(model_path, categories)

    if results is None:
        return None

    leaderboard_entry = {}
    for category_key, stats in results.items():
        leaderboard_entry[category_key] = stats

    return leaderboard_entry


def _save_leaderboard_entry(model_name: str, entry: dict):
    leaderboard_file = RESULT_DIR / f"leaderboard_{ROLLOUT_PREFIX}.json"
    leaderboard = {}
    if leaderboard_file.exists():
        try:
            with open(leaderboard_file, "r", encoding="utf-8") as f:
                leaderboard = json.load(f)
        except Exception as e:
            logging.error(f"Error reading {leaderboard_file}: {e}")

    if model_name not in leaderboard:
        leaderboard[model_name] = {}

    leaderboard[model_name].update(entry)

    with open(leaderboard_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)


def main(args):
    models = get_models(args.model)
    for model in models:
        logging.info(f"Generating for model: {model}")
        all_categories = get_test_categories(args.category)
        logging.info(f"Found {len(all_categories)} category(ies): {all_categories}")
        test_entries = collect_test_cases(
            model,
            categories=all_categories,
            prefix=ROLLOUT_PREFIX,
            trial_count=args.trial,
        )

        if len(test_entries) == 0:
            logging.info("All test cases have been completed. Skipping execution")
        else:
            logging.info(f"Collected {len(test_entries)} test case(s) for generation")
            asyncio.run(_generate_results_async(args, model, entries=test_entries))

        logging.info(f"Evaluating the model {model}...")
        model_name = model.replace("/", "_")
        model_path = RESULT_DIR / model_name
        if not model_path.exists():
            continue

        entry = _generate_leaderboard_entry(str(model_path), all_categories)
        if entry:
            _save_leaderboard_entry(model_name, entry)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model",
        type=str,
        help="Pretrained model name or local path to use for both rollout and sentence generation",
        required=True,
    )
    argparser.add_argument(
        "--category",
        type=str,
        help="Test dataset category or multiple categories to evaluate (e.g., math500, gsm8k)",
        required=True,
    )
    argparser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Number of trials per test case.",
    )
    argparser.add_argument(
        "--reasoning_tokens_budget",
        type=int,
        default=None,
        help="Maximum number of reasoning tokens to allocate per request.",
    )
    argparser.add_argument(
        "--answer_tokens_budget",
        type=int,
        default=None,
        help="Maximum number of answer tokens to allocate per request.",
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel worker threads. We recommend a value >= 16 for efficient batching.",
    )
    argparser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor-parallel size to use when spinning up each runtime.",
    )
    argparser.add_argument(
        "--max_total_tokens",
        type=int,
        default=None,
        help="Override the runtime KV-cache capacity (total tokens) per worker.",
    )
    argparser.add_argument(
        "--mem_fraction_static",
        type=float,
        default=0.65,
        help="GPU memory fraction reserved for static allocations per worker runtime.",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s [ThinkBrake] %(message)s ",
        datefmt="%m-%d %H:%M:%S",
    )

    args = parse_args()
    main(args)
