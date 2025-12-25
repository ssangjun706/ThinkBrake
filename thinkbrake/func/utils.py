import json
import random
import re

from typing import Optional, Union, Set
from thinkbrake.func.constants import DATA_DIR, RESULT_DIR, ROLLOUT_PREFIX
from thinkbrake.func.mapping import MODEL_MAPPING, CATEGORY_MAPPING
from thinkbrake.func.handler import BaseHandler


def extract_multiple_choice_answer(response: str) -> str:
    """
    Extract the multiple choice answer (A, B, C, D, etc.) from the response text.
    Uses common patterns found in model outputs for GPQA/MMLU benchmarks.
    """
    # Pattern priority: more explicit patterns first
    patterns = [
        # JSON-style: "answer": "D" or **answer**: "D"
        r'["\*]*answer["\*]*\s*[:=]\s*["\']?([A-Da-d])["\']?',
        # "The answer is D" or "Answer is D"
        r"(?:the\s+)?answer\s+is[:\s]*([A-Da-d])\b",
        # "Final answer: D"
        r"final\s+answer[:\s]*([A-Da-d])\b",
        # "Choice D" or "Option D"
        r"(?:choice|option)[:\s]*([A-Da-d])\b",
        # Standalone letter at the end (e.g., "D" or "d")
        r"\b([A-Da-d])\s*$",
    ]

    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            # Return the last match (usually the final answer)
            return matches[-1].upper()

    # Fallback: find the last standalone A/B/C/D in the response
    standalone_matches = re.findall(r"\b([A-Da-d])\b", response)
    if standalone_matches:
        return standalone_matches[-1].upper()

    return ""


def verify_multiple_choice(ground_truth: str, predicted: str) -> bool:
    """
    Verify if the predicted multiple choice answer matches the ground truth.
    """
    if not predicted:
        return False
    return ground_truth.upper().strip() == predicted.upper().strip()


def get_handler(pretrained_model_name_or_path: str) -> BaseHandler:
    """
    Factory function to get the appropriate model handler.

    Args:
        pretrained_model_name_or_path: Name or path of the model.

    Returns:
        An instance of a BaseHandler subclass.
    """
    handler_callable = MODEL_MAPPING[pretrained_model_name_or_path]
    handler: BaseHandler = handler_callable(
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    return handler


def get_test_case_id(test_case: dict) -> str:
    test_case_id = str(test_case["id"])
    if "sentence_idx" in test_case:
        test_case_id += f"_{test_case['sentence_idx']}"

    return test_case_id


def sort_key(item: dict) -> tuple:
    """
    Sort key function for test case ordering.

    Sorts by ID number and then sentence ID.
    """
    num = item["id"].split("_")[1]

    if num.isdigit():
        num = int(num)

    sentence_id = item.get("sentence_idx", 0)
    return (num, sentence_id)


def load_file(file_path: str) -> list[dict]:
    """
    Loads a JSONL file into a list of dictionaries.
    """
    try:
        data = []
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)

        return data
    except Exception:
        raise ValueError(f"File not found: {file_path}")


def get_parent_category(sub_category: str) -> str:
    """
    Finds the parent category (e.g., 'math', 'general') for a given sub-category.
    """
    parent_category = None
    for category in CATEGORY_MAPPING.keys():
        if sub_category in CATEGORY_MAPPING[category]:
            parent_category = category
            break

    return parent_category


def get_test_categories(categories: str) -> list[str]:
    """
    Parses category string (comma-separated or 'all') into a list of sub-categories.
    """
    if categories == "all":
        categories = list(CATEGORY_MAPPING.keys())
    else:
        categories = categories.split(",")

    parsed_categories = []
    for category in categories:
        if category in CATEGORY_MAPPING:
            parsed_categories.extend(CATEGORY_MAPPING[category])
        elif category in sum(CATEGORY_MAPPING.values(), []):
            parsed_categories.append(category)
        else:
            raise ValueError(f"Unknown category: {category}")

    return parsed_categories


def get_models(models: str) -> list[str]:
    """
    Parses model string (comma-separated or 'all') into a list of model names.
    """
    if models == "all":
        models = list(MODEL_MAPPING.keys())
    else:
        models = models.split(",")

    parsed_models = []
    for model in models:
        if model in MODEL_MAPPING:
            parsed_models.append(model)
        else:
            raise ValueError(f"Unknown model: {model}")

    return parsed_models


def get_thresholds(thresholds: Optional[str]) -> list[Optional[float]]:
    if thresholds is None:
        return [None]

    parsed_thresholds = []
    for threshold in thresholds.split(","):
        threshold = threshold.strip()
        try:
            parsed_thresholds.append(float(threshold))
        except ValueError:
            raise ValueError(f"Invalid threshold value: {threshold}")

    return parsed_thresholds


def get_test_entries_involved(
    model: str,
    category: str,
    prefix: str,
    threshold: Optional[float] = None,
) -> Set[str]:
    model_name = model.replace("/", "_")
    parent_category = get_parent_category(category)
    group_dir = RESULT_DIR / model_name / parent_category / prefix

    if threshold is not None:
        threshold_prefix = f"threshold_{threshold}"
        group_dir = group_dir / threshold_prefix

    file_path = group_dir / f"{category}_result.jsonl"

    if not file_path.exists():
        return set()

    test_entries = []
    with open(file_path, "r") as f:
        test_entries = [json.loads(line.strip()) for line in f]

    return test_entries


def collect_test_cases(
    model: str,
    categories: list[str],
    prefix: str,
    threshold: Optional[float] = None,
) -> list[dict]:
    """
    Collects test cases that need to be processed.
    Filters out cases that have already been completed.
    """
    data = []

    for category in categories:
        parent_category = get_parent_category(category)
        file_path = DATA_DIR / parent_category / f"{category}.jsonl"
        existing_ids = []

        existing_ids = [
            get_test_case_id(entry)
            for entry in get_test_entries_involved(
                model,
                category,
                prefix,
                threshold,
            )
        ]

        for idx, entry in enumerate(load_file(file_path)):
            if get_test_case_id(entry) not in existing_ids:
                data.append({"id": f"{category}_{idx}", **entry})

    random.shuffle(data)
    return data


def save_result(
    model: str,
    result: Union[dict, list[dict]],
    prefix: str,
    threshold: Optional[float] = None,
):
    model_escaped = model.replace("/", "_")
    model_result_dir = RESULT_DIR / model_escaped

    if isinstance(result, dict):
        result = [result]

    file_entries = {}
    for entry in result:
        test_category = entry["id"].split("_")[0]
        group_dir_name = get_parent_category(test_category)
        group_dir_path = model_result_dir / group_dir_name / prefix

        if threshold is not None:
            threshold_str = f"threshold_{threshold}"
            group_dir_path = group_dir_path / threshold_str

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


def split_sentence(
    items: list[dict],
    prefix="",
    eot_token="",
) -> list[dict]:
    test_cases = []
    suffix = prefix + eot_token
    for idx in range(len(items)):
        entire_response = items[idx].get("response", None)

        if entire_response is None:
            assert False, "No reasoning found to split."

        sentences = entire_response.split(eot_token)[0]
        sentences = sentences.rstrip().split(". ")

        concat_sents = []
        for i in range(len(sentences)):
            concat_sents.append(". ".join(sentences[: i + 1]))

        for i in range(len(concat_sents)):
            test_cases.append(
                {
                    "id": items[idx]["id"],
                    "category": items[idx]["category"],
                    "sentence_idx": i,
                    "problem": items[idx]["problem"],
                    "answer": items[idx]["answer"],
                    "assistant": concat_sents[i] + suffix,
                }
            )

    return test_cases
