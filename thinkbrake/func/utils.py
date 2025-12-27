import json
import random
import re
import glob
import ast
import math
from collections import defaultdict, Counter

from typing import Optional, Union, Set, List, Dict, Any
from thinkbrake.func.constants import DATA_DIR, RESULT_DIR
from thinkbrake.func.mapping import MODEL_MAPPING, CATEGORY_MAPPING
from thinkbrake.func.handler import BaseHandler


def calculate_pass_at_k(n, c, k):
    if n < k:
        return 1.0 if c > 0 else 0.0

    if c == n:
        return 1.0

    try:
        prob_fail = math.comb(n - c, k) / math.comb(n, k)
        return 1.0 - prob_fail
    except Exception:
        return 0.0


def _ast_get_name(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return _ast_get_name(node.value) + "." + node.attr
    return str(node)


def _ast_resolve(node):
    if isinstance(node, ast.Call):
        func_name = _ast_get_name(node.func)
        args_dict = {}
        for keyword in node.keywords:
            args_dict[keyword.arg] = _ast_resolve(keyword.value)
        return {func_name: args_dict}
    elif isinstance(node, ast.List):
        return [_ast_resolve(e) for e in node.elts]
    elif isinstance(node, ast.Dict):
        return {
            _ast_resolve(k): _ast_resolve(v) for k, v in zip(node.keys, node.values)
        }
    elif isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _ast_resolve(node.operand)
        if isinstance(operand, (int, float)):
            return -operand

    try:
        return ast.literal_eval(node)
    except:
        return None


def qwen_parse(input_str: Union[str, List, Dict]) -> List[Dict]:
    if isinstance(input_str, (list, dict)):
        return input_str if isinstance(input_str, list) else [input_str]

    results = []
    matches = re.findall(r"<tool_call>(.*?)</tool_call>", input_str, re.DOTALL)

    if matches:
        for match in matches:
            try:
                json_str = match.strip()
                parsed = json.loads(json_str)
                results.append(parsed)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_str)
                    results.append(parsed)
                except:
                    pass
    else:
        try:
            parsed = json.loads(input_str)
            if isinstance(parsed, list):
                results = parsed
            elif isinstance(parsed, dict):
                results = [parsed]
        except:
            pass

    return results


def default_parse(input_str: Union[str, List, Dict]) -> List[Dict]:
    if isinstance(input_str, (list, dict)):
        return input_str if isinstance(input_str, list) else [input_str]

    input_str = input_str.strip()
    input_str = input_str.strip("`").strip()
    if input_str.startswith("json\n"):
        input_str = input_str[5:]
    elif input_str.startswith("python\n"):
        input_str = input_str[7:]

    results = []
    try:
        if not input_str.startswith("["):
            input_str = "[" + input_str
        if not input_str.endswith("]"):
            input_str = input_str + "]"

        tree = ast.parse(input_str, mode="eval")
        evaluated = _ast_resolve(tree.body)

        if isinstance(evaluated, list):
            results = evaluated
        elif isinstance(evaluated, dict):
            results = [evaluated]

    except Exception:
        try:
            cleaned_str = input_str
            if cleaned_str.startswith('["') and not cleaned_str.endswith('"]'):
                cleaned_str = cleaned_str.replace('["', "[", 1)

            tree = ast.parse(cleaned_str, mode="eval")
            evaluated = _ast_resolve(tree.body)

            if isinstance(evaluated, list):
                results = evaluated
            elif isinstance(evaluated, dict):
                results = [evaluated]
        except Exception:
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = [parsed]
            except:
                pass

    final_results = []
    for item in results:
        if isinstance(item, str):
            try:
                tree = ast.parse(item, mode="eval")
                resolved = _ast_resolve(tree.body)
                if isinstance(resolved, dict):
                    final_results.append(resolved)
                else:
                    final_results.append(item)
            except:
                final_results.append(item)
        else:
            final_results.append(item)

    return final_results


def restructure_model_output(output):
    if not output:
        return output

    standardized = []
    for item in output:
        if isinstance(item, dict):
            if "name" in item and "arguments" in item and len(item) == 2:
                args = item["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                standardized.append({item["name"]: args})
            else:
                standardized.append(item)
    return standardized


def check_single_call(pred_name, pred_args, gt_item):
    if pred_name not in gt_item:
        return False

    gt_args_constraints = gt_item[pred_name]

    for arg, val in pred_args.items():
        if isinstance(val, str):
            val = val.strip()

        if arg not in gt_args_constraints:
            return False

        valid_options = gt_args_constraints[arg]
        if not isinstance(valid_options, list):
            valid_options = [valid_options]

        normalized_options = []
        for opt in valid_options:
            normalized_options.append(opt)
            if isinstance(opt, (int, float)):
                normalized_options.append(str(opt))

        if val not in valid_options and str(val) not in [str(o) for o in valid_options]:
            return False

    if len(pred_args) != len(gt_args_constraints):
        pass

    return True


def check_entry(prediction: List[Dict], ground_truth: List[Any]):
    if not prediction and not ground_truth:
        return True

    if len(prediction) != len(ground_truth):
        return False

    match_count = 0
    gt_copy = list(ground_truth)

    for pred_item in prediction:
        matched = False
        if not isinstance(pred_item, dict):
            continue

        pred_func_name = list(pred_item.keys())[0]
        pred_args = pred_item[pred_func_name]

        for idx, gt_item in enumerate(gt_copy):
            if check_single_call(pred_func_name, pred_args, gt_item):
                matched = True
                gt_copy.pop(idx)
                break

        if matched:
            match_count += 1

    return match_count == len(prediction) and len(gt_copy) == 0


def evaluate_bfcl_entry(entry: dict) -> bool:
    # Ground truth is now merged into answer field
    gt_answer = entry.get("answer")
    if gt_answer is None:
        # Check alternative key if necessary, or fail
        return False, None, False

    raw_result = entry.get("response")

    if "<tool_call>" in raw_result:
        parsed = qwen_parse(raw_result)
    else:
        parsed = default_parse(raw_result)

    reconstructed = restructure_model_output(parsed)
    result = check_entry(reconstructed, gt_answer)
    return reconstructed, gt_answer, result


def extract_multiple_choice_answer(response: str) -> str:
    patterns = [
        r'["\*]*answer["\*]*\s*[:=]\s*["\']?([A-Da-d])["\']?',
        r"(?:the\s+)?answer\s+is[:\s]*([A-Da-d])\b",
        r"final\s+answer[:\s]*([A-Da-d])\b",
        r"(?:choice|option)[:\s]*([A-Da-d])\b",
        r"\b([A-Da-d])\s*$",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    standalone_matches = re.findall(r"\b([A-Da-d])\b", response)
    if standalone_matches:
        return standalone_matches[-1].upper()

    return ""


def verify_multiple_choice(ground_truth: str, predicted: str) -> bool:
    if not predicted:
        return False
    return ground_truth.upper().strip() == predicted.upper().strip()


def get_handler(pretrained_model_name_or_path: str) -> BaseHandler:
    handler_callable = MODEL_MAPPING[pretrained_model_name_or_path]
    handler: BaseHandler = handler_callable(
        pretrained_model_name_or_path=pretrained_model_name_or_path
    )
    return handler


def get_test_case_id(test_case: dict) -> str:
    test_case_id = f"id_{test_case["id"]}_trial_{test_case["trial"]}"
    return test_case_id


def sort_key(item: dict) -> tuple:
    try:
        parts = item["id"].rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            num = int(parts[1])
        else:
            num = item["id"]
    except Exception:
        num = item["id"]

    sentence_id = item.get("sentence_idx", 0)
    trial = item.get("trial", 1)
    return (num, sentence_id, trial)


def load_file(file_path: str) -> list[dict]:
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
    parent_category = None
    for category in CATEGORY_MAPPING.keys():
        if sub_category in CATEGORY_MAPPING[category]:
            parent_category = category
            break

    return parent_category


def get_test_categories(categories: str) -> list[str]:
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
    trial_count: int = 1,
) -> list[dict]:
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

        for idx, base_entry in enumerate(load_file(file_path)):
            for t in range(1, trial_count + 1):
                candidate = base_entry.copy()
                candidate["trial"] = t
                if get_test_case_id(candidate) not in existing_ids:
                    data.append(candidate)

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
                new_entry = {}
                new_entry["id"] = entry["id"]

                if "sentence_idx" in entry:
                    new_entry["sentence_idx"] = entry["sentence_idx"]

                new_entry["trial"] = entry["trial"]
                new_entry.update(entry)
                f.write(json.dumps(new_entry) + "\n")
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


def calculate_metrics(problems, total_entries=0, total_tokens=0):
    num_problems = len(problems.keys())
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

    max_trials = 0
    for trials in problems.values():
        max_trials = max(max_trials, len(trials))

    ks_to_track = [1, 2, 4, 5, 8, 10, 16, 32, 64, 100]
    ks_to_track = [k for k in ks_to_track if k <= max_trials]
    if max_trials not in ks_to_track and max_trials > 0:
        ks_to_track.append(max_trials)
    ks_to_track.sort()

    pass_at_k_sums = defaultdict(float)
    avg_at_n_sums = defaultdict(float)
    avg_at_n_counts = defaultdict(int)

    for key, trials in problems.items():
        n = len(trials)
        c = sum(1 for t in trials if t["is_correct"])

        # 1. Average Accuracy (Pass@1 effectively, averaged over problems)
        sum_accuracy += c / n

        # 2. Majority Vote
        preds = [str(t["predicted"]) for t in trials if t["predicted"]]

        if preds:
            counter = Counter(preds)
            most_common = counter.most_common(1)
            majority_pred = most_common[0][0]

            # Check if majority_pred corresponds to a correct trial
            majority_is_correct = False
            for t in trials:
                # We compare strings here.
                if t["predicted"] == majority_pred and t["is_correct"]:
                    majority_is_correct = True
                    break

            if majority_is_correct:
                sum_majority += 1.0

        # 3. Pass@k
        for k in ks_to_track:
            pass_at_k_sums[k] += calculate_pass_at_k(n, c, k)

        # 4. Avg@n (mean accuracy over first n trials)
        for k in ks_to_track:
            if n >= k:
                first_k_correct = sum(1 for t in trials[:k] if t["is_correct"])
                avg_at_n_sums[k] += first_k_correct / k
                avg_at_n_counts[k] += 1

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
        "avg@n": {
            k: (
                (avg_at_n_sums[k] / avg_at_n_counts[k] * 100)
                if avg_at_n_counts[k] > 0
                else 0.0
            )
            for k in ks_to_track
        },
    }
