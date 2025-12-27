import json
import os
import re
import ast
from datasets import load_dataset

# Constants
OUTPUT_DIR = "/home/work/ThinkBrake/thinkbrake/data/tool"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "api-bank.jsonl")


def parse_history_tools(history_text):
    """
    Parse the history text to find tool definitions returned by ToolSearcher.
    Format in history:
    API-Request: [ToolSearcher(...)]->{'api_name': 'ToolSearcher', 'input': ..., 'output': {'name': 'ToolName', ...}, ...}
    """
    tools = []
    parts = history_text.split("->")
    for part in parts[1:]:
        try:
            brace_count = 0
            json_str = ""
            started = False
            for char in part:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1

                if started:
                    json_str += char
                    if brace_count == 0:
                        break

            if json_str:
                output_obj = ast.literal_eval(json_str)
                if isinstance(output_obj, dict) and "output" in output_obj:
                    tool_def = output_obj["output"]
                    if (
                        isinstance(tool_def, dict)
                        and "name" in tool_def
                        and "input_parameters" in tool_def
                    ):
                        tools.append(tool_def)
        except Exception as e:
            pass

    return tools


def convert_tool_def_to_bfcl(tool_def):
    params = tool_def.get("parameters", tool_def.get("input_parameters", {}))
    return {
        "name": tool_def.get("name", tool_def.get("apiCode")),
        "description": tool_def.get("description"),
        "parameters": {
            "type": "dict",
            "properties": params,
            "required": list(params.keys()),
        },
    }


def parse_api_request_string(request_str):
    """
    Parse 'API-Request: [Tool(arg='val')]' into BFCL GT format.
    BFCL GT format: List of dicts, e.g. [{'Tool': {'arg': ['val']}}]
    """
    try:
        if "API-Request:" in request_str:
            request_str = request_str.split("API-Request:")[1].strip()

        if request_str.startswith("[") and request_str.endswith("]"):
            request_str = request_str[1:-1]

        calls = []
        matches = re.finditer(r"([a-zA-Z0-9_]+)\((.*?)\)", request_str)

        for match in matches:
            func_name = match.group(1)
            args_str = match.group(2)

            args = {}
            arg_matches = re.finditer(
                r"([a-zA-Z0-9_]+)=(['\"])(.*?)\2|([a-zA-Z0-9_]+)=([0-9.]+)", args_str
            )

            for am in arg_matches:
                if am.group(1):  # Quoted string
                    key = am.group(1)
                    val = am.group(3)
                else:  # Number
                    key = am.group(4)
                    val = am.group(5)

                args[key] = [val]

            calls.append({func_name: args})

        return calls
    except Exception:
        return []


def convert_to_bfcl_format(entry, index, level="lev3"):
    input_text = entry.get("input", "")

    # Try different parsing strategies based on input format
    try:
        user_part = ""
        bfcl_tools = []

        # Level 3 Style (and some Level 2)
        if "\nUser:" in input_text:
            json_part = input_text.split("\nUser:")[0]
            user_part = input_text.split("\nUser:")[1].split("\nGenerate")[0].strip()

            try:
                tool_def = json.loads(json_part)
                bfcl_tools = [convert_tool_def_to_bfcl(tool_def)]
            except:
                pass

            extracted_tools = parse_history_tools(user_part)
            for t in extracted_tools:
                if not any(existing["name"] == t["name"] for existing in bfcl_tools):
                    bfcl_tools.append(convert_tool_def_to_bfcl(t))

        # Fallback / Level 1 Style (Instruction based)
        else:
            # Basic extraction for Level 1 which might differ
            user_part = input_text.replace("Generate an API request", "").strip()

            # Level 1 usually has context in 'instruction' field if 'input' is simple
            if "instruction" in entry:
                instruction = entry["instruction"]
                if "API descriptions:" in instruction:
                    desc_part = instruction.split("API descriptions:")[1].strip()
                    try:
                        tool_def = json.loads(desc_part)
                        bfcl_tools = [convert_tool_def_to_bfcl(tool_def)]
                    except:
                        pass
                if "Input:" in instruction:
                    # Sometimes input has the dialogue
                    # We can try to extract more context if needed
                    pass

        if not user_part:
            user_part = entry.get("instruction", "")

        # Common Output Parsing
        raw_answer = entry.get("output", "").strip()
        if not raw_answer and "expected_output" in entry:
            raw_answer = entry["expected_output"]

        # Parse ground truth
        ground_truth = parse_api_request_string(raw_answer)

        # Unique ID generation (global index passed in)
        unique_id = f"api-bank_{index}"

        bfcl_entry = {
            "id": unique_id,
            "category": "tool",
            "sub_category": f"api-bank-{level}",
            "answer": ground_truth,
            "problem": [{"role": "user", "content": user_part}],
            "function": bfcl_tools,
        }

        # Filter out empty answers or problems to ensure quality (optional but good)
        if not ground_truth:
            return None

        return bfcl_entry

    except Exception as e:
        # print(f"Skipping entry {index} due to parse error: {e}")
        return None


def main():
    print("Preparing API-Bank dataset (Combined Levels)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_source_config = [
        {"name": "lev1", "file": "test-data/level-1-api.json"},
        {"name": "lev2", "file": "test-data/level-2-api.json"},
        {"name": "lev3", "file": "test-data/level-3-batch-inf.json"},
    ]

    total_processed = 0
    global_index = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for config in data_source_config:
            level_name = config["name"]
            file_path = config["file"]

            print(f"Processing {level_name} from {file_path}...")

            try:
                # Load specific file
                dataset = load_dataset(
                    "liminghao1630/API-Bank",
                    data_files={"data": file_path},
                    split="data",
                )
                print(f"  Loaded {len(dataset)} examples for {level_name}.")

                count = 0
                for entry in dataset:
                    bfcl_entry = convert_to_bfcl_format(
                        entry, global_index, level=level_name
                    )
                    if bfcl_entry:
                        f.write(json.dumps(bfcl_entry, ensure_ascii=False) + "\n")
                        count += 1
                        global_index += 1

                print(f"  Saved {count} examples for {level_name}.")
                total_processed += count

            except Exception as e:
                print(f"  Failed to process {level_name}: {e}")

    print(f"Successfully processed {total_processed} total examples.")
    print(f"Saved dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
