from thinkbrake.func.handler import (
    Qwen3Handler,
    DeepSeekHandler,
    # GptOssHandler,
    Phi4Handler,
)

MODEL_MAPPING = {
    "Qwen/Qwen3-4B-Thinking-2507": Qwen3Handler,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": DeepSeekHandler,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": DeepSeekHandler,
    "Qwen/Qwen3-4B": Qwen3Handler,
    "Qwen/Qwen3-14B": Qwen3Handler,
    "Qwen/Qwen3-32B": Qwen3Handler,
    "microsoft/phi-4-reasoning": Phi4Handler,
    # "openai/gpt-oss-20b": GptOssHandler,
}

CATEGORY_MAPPING = {
    "math": ["gsm8k", "math500", "aime2024", "aime2025"],
    "general": ["gpqa-diamond"],  # , "mmlu-redux"],
    # "tool": ["bfcl-v1", "bfcl-v2"],  # , "meta-tool"],
}
