import json
import os
from datasets import load_dataset


def main():
    # Load dataset
    print("Loading ARC-Challenge dataset...")
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    output_dir = "/home/work/ThinkBrake/thinkbrake/data/general"
    output_file = os.path.join(output_dir, "arc-challenge.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(ds)} examples...")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            question = item["question"]
            choices = item["choices"]

            choice_text_list = []
            labels = choices.get("label", [])
            texts = choices.get("text", [])

            label_map = {"1": "A", "2": "B", "3": "C", "4": "D"}

            # Combine labels and texts
            for label, text in zip(labels, texts):
                norm_label = label_map.get(label, label)
                choice_text_list.append(f"{norm_label}. {text}")

            formatted_choices = "\n".join(choice_text_list)
            problem_text = f"{question}\n\n{formatted_choices}"

            # Normalize answer key
            answer_key = item["answerKey"]
            answer_key = label_map.get(answer_key, answer_key)

            record = {
                "id": f"arc-challenge_{i}",
                "category": "general",
                "answer": answer_key,
                "problem": problem_text,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(ds)} items to {output_file}")


if __name__ == "__main__":
    main()
