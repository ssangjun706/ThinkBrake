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
    calculate_pass_at_k,
    calculate_metrics,
)


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

    finally:
        write_queue.put(None)
        writer_thread.join()
        handler.shutdown_local_server()


def _evaluate_jsonl_file(file_path: str, sub_category: str = None) -> dict:
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
                    predicted = str(parsed_pred)
                    is_correct = verify(ground_truth, parsed_pred)
                elif parent_category == "tool":
                    if sub_category in ["bfcl-v1", "bfcl-v2", "api-bank"]:
                        pred_obj, ground_truth, is_correct = evaluate_bfcl_entry(item)
                        predicted = str(pred_obj)
                    else:
                        ground_truth = None
                        predicted = None
                        pass
                else:
                    raise ValueError(f"Unknown category found: {parent_category}")

                problem_id = item.get("id")

                problems[problem_id].append(
                    {
                        "predicted": predicted,
                        "is_correct": is_correct,
                        "ground_truth": ground_truth,
                    }
                )

        except Exception as e:
            logging.error(f"Error evaluating the file {file_path}: {e}")
            return None

    return calculate_metrics(
        problems, total_entries=total_entries, total_tokens=total_tokens
    )


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
                "avg@n": eval_result["avg@n"],
            }

            logging.info("")
            logging.info(f"Category: {sub_category}")
            logging.info(f"- Avg Acc (Pass@1) : {eval_result['accuracy']:.2f}%")
            logging.info(
                f"- Majority Vote    : {eval_result['majority_accuracy']:.2f}%"
            )

            # Log Pass@K (only max k)
            if eval_result["pass@k"]:
                max_k = max(eval_result["pass@k"].keys())
                logging.info(
                    f"- Pass@{max_k}         : {eval_result['pass@k'][max_k]:.2f}%"
                )

            # Log Avg@N (only max n)
            if eval_result["avg@n"]:
                max_n = max(eval_result["avg@n"].keys())
                logging.info(
                    f"- Avg@{max_n}          : {eval_result['avg@n'][max_n]:.2f}%"
                )

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
