import json
import os
from datasets import load_dataset


def main():
    # Load dataset
    print("Loading Omni-MATH dataset...")
    try:
        # Load the dataset. Usually prompts are often in 'test' split for benchmarks.
        # If 'test' is not found, we might need to check available splits.
        # Let's try 'test' first as per user's implication of "like arc-challenge"
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    except Exception as e:
        print(f"Error loading dataset with split='test': {e}")
        try:
            # Fallback to 'train' if 'test' doesn't exist, or print available splits
            ds_dict = load_dataset("KbsdJames/Omni-MATH")
            print(f"Available splits: {ds_dict.keys()}")
            # If there's only one split, use it.
            if len(ds_dict.keys()) == 1:
                split_name = list(ds_dict.keys())[0]
                ds = ds_dict[split_name]
            else:
                return
        except Exception as e2:
            print(f"Error loading dataset generic: {e2}")
            return

    # Define output path
    output_dir = "/home/work/ThinkBrake/thinkbrake/data/math"
    output_file = os.path.join(output_dir, "omni-math.jsonl")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(ds)} examples...")

    # Inspect first item to identify column names
    if len(ds) > 0:
        print("First item keys:", ds[0].keys())
        first_item = ds[0]

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            # Attempt to find relevant fields.
            # Math datasets often have 'problem', 'question' -> problem
            # 'solution', 'answer' -> answer
            # 'difficulty' -> difficulty

            problem = item.get("problem") or item.get("question")
            answer = item.get("answer") or item.get("solution")
            difficulty = item.get("difficulty", "")  # Default to empty if missing

            if problem is None or answer is None:
                print(
                    f"Skipping item {i} due to missing fields. Keys found: {item.keys()}"
                )
                continue

            record = {
                "id": f"omni-math_{i}",
                "category": "math",
                "problem": problem,
                "answer": answer,
                "difficulty": difficulty,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(ds)} items to {output_file}")


if __name__ == "__main__":
    main()
