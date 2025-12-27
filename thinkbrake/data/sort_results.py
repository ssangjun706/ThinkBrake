import argparse
import json
import os
from pathlib import Path


def sort_key(item):
    # Try to extract the numeric part from the end of the id for natural sort behavior
    id_str = item.get("id", "")
    try:
        # Assuming format like "dataset_name_123"
        parts = id_str.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            id_prefix = parts[0]
            id_num = int(parts[1])
        else:
            # Fallback to string sort if pattern doesn't match
            id_prefix = id_str
            id_num = -1
    except:
        id_prefix = id_str
        id_num = -1

    # sentence_idx default to -1 (so they come first? or 0? valid idx starts at 0. usually doesn't matter as long as consistent)
    # utils.py uses default 0. Let's use -1 to distinguish missing.
    sentence_idx = item.get("sentence_idx", -1)
    trial = item.get("trial", 1)

    return (id_prefix, id_num, sentence_idx, trial)


def sort_file(file_path):
    print(f"Processing {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line in {file_path}: {e}")

        # Sort
        data.sort(key=sort_key)

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                # Reorder keys for readability: id, sentence_idx, trial, then others
                new_item = {"id": item.get("id")}

                if "sentence_idx" in item:
                    new_item["sentence_idx"] = item["sentence_idx"]

                if "trial" in item:
                    new_item["trial"] = item["trial"]

                # Add remaining keys
                for k, v in item.items():
                    if k not in ["id", "sentence_idx", "trial"]:
                        new_item[k] = v

                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sort result JSONL files by id, sentence_idx, and trial."
    )
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="outputs/Qwen_Qwen3-4B-Thinking-2507/math/oracle",
        help="Directory to search for .jsonl files",
    )
    args = parser.parse_args()

    target_path = Path(args.target_dir)
    if not target_path.exists():
        # If relative path not found, try adding project root prefix if running from elsewhere?
        # Assuming run from workspace root
        print(f"Path {target_path} does not exist.")
        return

    # Look for jsonl files. We target *_result.jsonl specifically to be safe, or just all .jsonl in outputs?
    # User said "some result json". Let's target all .jsonl in the target dir recursively.
    files = list(target_path.rglob("*.jsonl"))

    if not files:
        print(f"No JSONL files found in {target_path}")
        return

    print(f"Found {len(files)} JSONL files to process in number.")

    for file_path in files:
        sort_file(file_path)


if __name__ == "__main__":
    main()
