import json
import os
from datasets import load_dataset


def main():
    print("Loading DeepScaleR dataset...")
    try:
        ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    output_dir = "/home/work/ThinkBrake/thinkbrake/data/math"
    output_file = os.path.join(output_dir, "deepscale-r.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(ds)} examples...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            problem = item.get("problem")
            answer = item.get("answer")

            if problem is None or answer is None:
                continue

            record = {
                "id": f"deepscale-r_{i}",
                "category": "math",
                "problem": problem,
                "answer": answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(ds)} items to {output_file}")


if __name__ == "__main__":
    main()
