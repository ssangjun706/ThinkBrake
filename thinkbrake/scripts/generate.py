import argparse
import logging
import queue
import threading
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from tqdm import tqdm
from thinkbrake.func.constants import THINKBRAKE_PREFIX
from thinkbrake.func.handler import BaseHandler
from thinkbrake.func.utils import (
    collect_test_cases,
    get_handler,
    get_models,
    get_test_categories,
    get_thresholds,
    save_result,
    get_test_case_id,
)


async def _process_entry_async(
    handler: BaseHandler,
    entry: dict,
    threshold: float,
    reasoning_tokens_budget: Optional[int],
    answer_tokens_budget: Optional[int],
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        result = await handler.inference_async(
            entry,
            reasoning_tokens_budget=reasoning_tokens_budget,
            answer_tokens_budget=answer_tokens_budget,
            threshold=threshold,
        )

    return {**entry, **result}


async def _generate_results_async(args, entries: List[dict]) -> None:
    loop = asyncio.get_running_loop()
    pool_size = args.num_workers + 4
    loop.set_default_executor(ThreadPoolExecutor(max_workers=pool_size))

    handler: BaseHandler = get_handler(args.model)
    handler.spin_up_local_server(
        tensor_parallel_size=args.tensor_parallel_size,
        max_total_tokens=args.max_total_tokens,
        mem_fraction_static=args.mem_fraction_static,
    )
    handler.reasoning_chunk_size = args.reasoning_chunk_size

    num_workers = args.num_workers
    write_queue = queue.Queue()

    def _writer():
        while True:
            item = write_queue.get()
            if item is None:
                break

            save_result(
                model=args.model,
                result=item,
                prefix=THINKBRAKE_PREFIX,
                threshold=args.threshold,
            )
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
                        _process_entry_async(
                            handler,
                            entry,
                            args.threshold,
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
                        raise SystemExit(f"Fatal error in entry {entry_id}: {e}")

                    pbar.update()

    finally:
        write_queue.put(None)
        writer_thread.join()
        handler.shutdown_local_server()


def main(args):
    all_categories = get_test_categories(args.category)
    logging.info(f"Found {len(all_categories)} category(ies): {all_categories}")
    test_entries = collect_test_cases(
        args.model,
        categories=all_categories,
        threshold=args.threshold,
        prefix=THINKBRAKE_PREFIX,
        trial_count=args.trial,
    )

    if len(test_entries) == 0:
        logging.info("All test cases have been completed. Skipping execution")
    else:
        logging.info(f"Collected {len(test_entries)} test case(s) for generation")
        asyncio.run(_generate_results_async(args, entries=test_entries))


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
        "--reasoning_tokens_budget",
        type=int,
        default=None,
        help="Maximum number of reasoning tokens to allocate per request.",
    )
    argparser.add_argument(
        "--reasoning_chunk_size",
        type=int,
        default=None,
        help="Cap on reasoning tokens decoded per chunk before yielding.",
    )
    argparser.add_argument(
        "--answer_tokens_budget",
        type=int,
        default=4096,
        help="Maximum number of answer tokens to allocate per request.",
    )
    argparser.add_argument(
        "--threshold",
        type=str,
        default=None,
        help="Comma-separated threshold values for early stopping (e.g., '0.1,0.5,0.9'). If not specified, runs without threshold.",
    )
    argparser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Number of trials per test case.",
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
    thresholds = get_thresholds(args.threshold)
    for model in models:
        for threshold in thresholds:
            logging.info(f"Generating for model: {model}, threshold: {threshold}")
            args.model = model
            args.threshold = threshold
            main(args)
