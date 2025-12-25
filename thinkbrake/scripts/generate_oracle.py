import argparse
import json
import logging
import queue
import threading
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

from tqdm import tqdm
from thinkbrake.func.constants import (
    ORACLE_PREFIX,
    RESULT_DIR,
    ROLLOUT_PREFIX,
)
from thinkbrake.func.handler import BaseHandler
from thinkbrake.func.utils import (
    get_handler,
    get_models,
    get_parent_category,
    get_test_case_id,
    get_test_categories,
    load_file,
    sort_key,
    split_sentence,
)


def save_result(
    model: str,
    result: Union[dict, list[dict]],
):
    model_result_dir = RESULT_DIR / model.replace("/", "_")

    if isinstance(result, dict):
        result = [result]

    file_entries = {}
    for entry in result:
        test_category = entry["id"].split("_")[0]
        group_dir_path = (
            model_result_dir / get_parent_category(test_category) / ORACLE_PREFIX
        )
        group_dir_path.mkdir(parents=True, exist_ok=True)

        file_path = group_dir_path / f"{test_category}_result.jsonl"
        file_entries.setdefault(file_path, []).append(entry)

    for file_path, entries in file_entries.items():
        existing_entries = {}
        if file_path.exists():
            existing_entries = {
                get_test_case_id(entry): entry for entry in load_file(file_path)
            }

        for entry in entries:
            existing_entries[get_test_case_id(entry)] = entry

        sorted_entries = sorted(existing_entries.values(), key=sort_key)
        with open(file_path, "w") as f:
            for entry in sorted_entries:
                content = json.dumps(entry) + "\n"
                f.write(content)
            f.flush()


async def _process_entry_async(
    handler: BaseHandler,
    entry: dict,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        result = await handler.inference_oracle(entry)

    return {**entry, **result}


async def _generate_results_async(
    args,
    entries: List[dict],
    handler: BaseHandler,
) -> None:
    loop = asyncio.get_running_loop()
    pool_size = args.num_workers + 4
    loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

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

            save_result(args.model, item)
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
            desc=f"Generating results for {args.model}",
        ) as pbar:

            while ready_queue or in_flight:
                while ready_queue:
                    entry_id = ready_queue.popleft()
                    entry = id_to_entry[entry_id]

                    task = asyncio.create_task(
                        _process_entry_async(handler, entry, sem)
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
                        raise SystemExit(f"Fatal error in entry {entry_id}: {e}")

                    pbar.update()

    finally:
        write_queue.put(None)
        writer_thread.join()
        handler.shutdown_local_server()


def get_oracle_entries_involved(
    model: str,
    category: str,
) -> list[dict]:
    model_dir = RESULT_DIR / model.replace("/", "_")
    category_dir = model_dir / get_parent_category(category) / ORACLE_PREFIX
    file_path = category_dir / f"{category}_result.jsonl"

    if not file_path.exists():
        return []

    test_entries = []
    with open(file_path, "r") as f:
        test_entries = [json.loads(line.strip()) for line in f]

    return test_entries


def collect_rollout_cases(
    model: str,
    categories: list[str],
    prefix: str,
    eot_token: str,
) -> list[dict]:
    data = []

    for category in categories:
        parent_category = get_parent_category(category)
        model_dir = RESULT_DIR / model.replace("/", "_")
        category_dir = model_dir / parent_category / ROLLOUT_PREFIX
        file_path = category_dir / f"{category}_result.jsonl"

        existing_entries = get_oracle_entries_involved(model, category)
        existing_ids = [get_test_case_id(entry) for entry in existing_entries]

        entries = split_sentence(
            load_file(file_path),
            prefix=prefix,
            eot_token=eot_token,
        )

        filtered_entries = [
            entry for entry in entries if get_test_case_id(entry) not in existing_ids
        ]

        if len(filtered_entries) > 0:
            logging.info(
                f"Category {category}: {len(entries)} total entries, "
                f"{len(existing_ids)} already processed, "
                f"{len(filtered_entries)} remaining to process"
            )

        data.extend(filtered_entries)

    return data


def main(args):
    all_categories = get_test_categories(args.category)
    logging.info(f"Found {len(all_categories)} category(ies): {all_categories}")

    handler = get_handler(args.model)
    test_entries = collect_rollout_cases(
        args.model,
        categories=all_categories,
        prefix=handler.prefix_token,
        eot_token=handler.eot_token,
    )

    if len(test_entries) == 0:
        logging.info("All test cases have been completed. Skipping execution")
    else:
        logging.info(f"Collected {len(test_entries)} test case(s) for generation")

    asyncio.run(_generate_results_async(args, entries=test_entries, handler=handler))


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
        "--threshold",
        type=str,
        default=None,
        help="Comma-separated threshold values for early stopping (e.g., '0.1,0.5,0.9'). If not specified, runs without threshold.",
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=100,
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
        default=0.8,
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
    models = get_models(args.model)
    for model in models:
        logging.info(f"Generating for model: {model}")
        args.model = model
        main(args)
