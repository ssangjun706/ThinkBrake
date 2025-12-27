import json
import os
import random
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Sample GSM8K train data for validation."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to select for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="thinkbrake/data/math/gsm8k-val.jsonl",
        help="Path to save the output JSONL file",
    )
    args = parser.parse_args()

    print("Loading GSM8K dataset (train split)...")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    random.seed(args.seed)

    total_samples = len(dataset)
    if args.num_samples > total_samples:
        print(
            f"Warning: requested {args.num_samples} samples, but only {total_samples} are available. Using all."
        )
        args.num_samples = total_samples

    indices = list(range(total_samples))
    random.shuffle(indices)
    selected_indices = indices[: args.num_samples]

    selected_data = dataset.select(selected_indices)

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(
        f"Processing and saving {len(selected_data)} samples to {args.output_path}..."
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(selected_data):
            entry = {
                "id": f"gsm8k-val_{i}",
                "category": "math",
                "answer": item["answer"].split("####")[-1].strip(),
                "problem": item["question"],
            }
            f.write(json.dumps(entry) + "\n")

    print("Success!")


if __name__ == "__main__":
    main()
