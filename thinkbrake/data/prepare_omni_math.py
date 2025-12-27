import json
import os
from datasets import load_dataset


def main():
    print("Loading Omni-MATH dataset...")
    try:
        ds = load_dataset("KbsdJames/Omni-MATH", split="test")
    except Exception as e:
        return

    output_dir = "/home/work/ThinkBrake/thinkbrake/data/math"
    output_file = os.path.join(output_dir, "omni-math.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    skipped = 0
    processed = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            problem = item.get("problem") or item.get("question")
            answer = item.get("answer") or item.get("solution")
            difficulty = item.get("difficulty", "")

            if float(difficulty) > 5.0:
                skipped += 1
                continue

            if problem is None or answer is None:
                continue

            processed += 1
            record = {
                "id": f"omni-math_{i}",
                "category": "math",
                "problem": problem,
                "answer": answer,
                "difficulty": difficulty,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully saved {processed} items to {output_file}")
    print(f"Skipped {skipped} items due to difficulty > 5.0")


if __name__ == "__main__":
    main()
