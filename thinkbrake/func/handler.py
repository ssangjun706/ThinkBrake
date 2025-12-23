import sglang as sgl
import json
import asyncio

from datetime import datetime
from typing import Optional, final, Any


@sgl.function
def run_thinkbrake_step(
    s,
    prompt: str,
    threshold: Optional[float],
    eot_token: str,
    eos_token: str,
    prefix_token: Optional[str] = "",
    suffix_token: Optional[str] = "",
    reasoning_tokens_budget: Optional[int] = 16384,
    answer_tokens_budget: Optional[int] = 2048,
    max_total_tokens: Optional[int] = None,
    sampling_params: Optional[dict] = None,
):
    """
    SGLang function to run a single step of ThinkBrake reasoning.

    This function generates the reasoning trace (thought process) and monitors the
    token probabilities to decide when to stop thinking (braking) and generate the final answer.

    Args:
        s: The SGLang state object.
        prompt: The input prompt.
        threshold: Probability threshold for early stopping of reasoning.
        eot_token: End-of-thought token.
        eos_token: End-of-sentence token.
        prefix_token: Token to prepend to the closing sequence.
        suffix_token: Token to append to the closing sequence.
        reasoning_tokens_budget: Max tokens for reasoning.
        answer_tokens_budget: Max tokens for the final answer.
        max_total_tokens: Hard cap on total tokens (reasoning + answer).
        sampling_params: Sampling parameters for generation.
    """
    s += prompt
    reasoning_budget = (
        reasoning_tokens_budget if reasoning_tokens_budget is not None else 16384
    )
    answer_budget = answer_tokens_budget if answer_tokens_budget is not None else 2048

    # Calculate effective total budget
    total_budget = reasoning_budget + answer_budget
    if max_total_tokens is not None:
        total_budget = min(total_budget, max_total_tokens)

    # Track total tokens used across reasoning
    total_reasoning_tokens_used = 0
    remaining_tokens = reasoning_budget

    threshold = threshold if threshold is not None else 0.0
    gen_sampling_params = dict(sampling_params or {})
    probe_sampling_params = dict(gen_sampling_params)
    closing_token = (prefix_token or "") + eot_token
    closing_sequence = closing_token + (suffix_token or "")

    while True:
        # Calculate chunk budget respecting total limit
        # Ensure we leave room for answer_budget
        max_allowed_reasoning = (
            total_budget - answer_budget - total_reasoning_tokens_used
        )
        if max_allowed_reasoning <= 0:
            # No more room for reasoning, force stop
            s += closing_sequence
            break

        chunk_budget = (
            min(remaining_tokens, max_allowed_reasoning)
            if remaining_tokens > 0
            else min(1024, max_allowed_reasoning)
        )
        if chunk_budget <= 0:
            s += closing_sequence
            break

        s += sgl.gen(
            name="main_chunk",
            max_tokens=chunk_budget,
            stop=[".\n\n", eot_token],
            **gen_sampling_params,
        )
        main_meta = s.get_meta_info("main_chunk")
        finish_reason = main_meta.get("finish_reason") or {}
        finish_type = finish_reason.get("type")
        original_token = finish_reason.get("matched")
        tokens_generated = main_meta["completion_tokens"] + 1
        remaining_tokens -= tokens_generated
        total_reasoning_tokens_used += tokens_generated

        if (
            original_token == eot_token
        ):  # Model generated the [END_OF_THINK] token explicitly
            s += eot_token
            break

        if not original_token:
            if finish_type == "length":
                continue

            s += closing_sequence
            break

        # Speculatively fork to check probabilities
        forks = s.fork(2)

        # Branch 1: Continue with original token
        forks[0] += original_token
        forks[0] += sgl.gen(
            name="check_next_token",
            max_tokens=1,
            logprob_start_len=-1,
            return_logprob=True,
            top_logprobs_num=1,
            **probe_sampling_params,
        )

        forks[1] += closing_token
        forks[1] += sgl.gen(
            name="check_think_token",
            max_tokens=0,
            logprob_start_len=0,
            return_logprob=True,
            top_logprobs_num=1,
            **probe_sampling_params,
        )

        # Compare logprobs
        next_meta = forks[0].get_meta_info("check_next_token")
        next_tokens = next_meta.get("output_token_logprobs")
        top_token_logprob, _, _ = next_tokens[-1]

        eot_meta = forks[1].get_meta_info("check_think_token")
        eot_tokens = eot_meta.get("input_token_logprobs")
        eot_logprob, _, _ = eot_tokens[-1]

        # Decision logic for braking
        if top_token_logprob - eot_logprob <= threshold:
            if suffix_token:
                remaining_tokens -= 1
                total_reasoning_tokens_used += 1

            s += closing_sequence
            break
        else:
            s += original_token

    # Generate final answer with remaining budget
    remaining_total_budget = total_budget - total_reasoning_tokens_used
    effective_answer_budget = (
        min(answer_budget, remaining_total_budget) if remaining_total_budget > 0 else 1
    )

    s += sgl.gen(
        name="final_answer",
        max_tokens=effective_answer_budget,
        stop=[eos_token],
        **gen_sampling_params,
    )

    return s


@sgl.function
def run_rollout(
    s,
    prompt: str,
    eos_token: str,
    max_tokens: Optional[int] = 16384,
    sampling_params: Optional[dict] = None,
):
    s += prompt
    s += sgl.gen(
        name="final_answer",
        max_tokens=max_tokens,
        stop=[eos_token],
        **sampling_params,
    )

    return s


class BaseHandler:
    def __init__(self, pretrained_model_name_or_path: str):
        self.client: Optional[sgl.Runtime] = None
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.runtime: Optional[sgl.Runtime] = None
        self.max_total_tokens: Optional[int] = None

        self.eos_token = ""
        self.prefix_token = ""
        self.suffix_token = ""
        self.eot_token = ""
        self.sampling_params = {}

    @final
    def spin_up_local_server(
        self,
        tensor_parallel_size: int = 1,
        max_total_tokens: Optional[int] = None,
        mem_fraction_static: float = 0.75,
        **runtime_kwargs,
    ):
        runtime_config = {
            "model_path": self.pretrained_model_name_or_path,
            "tokenizer_path": self.pretrained_model_name_or_path,
            "tp_size": tensor_parallel_size,
            "mem_fraction_static": mem_fraction_static,
            "schedule_policy": "lpm",
        }

        if max_total_tokens is not None:
            runtime_config["max_total_tokens"] = max_total_tokens
            self.max_total_tokens = max_total_tokens

        runtime_config.update(runtime_kwargs)

        if self.runtime is not None:
            self.runtime.close()

        self.runtime: sgl.Runtime = sgl.Runtime(**runtime_config)
        sgl.set_default_backend(self.runtime)

    @final
    def shutdown_local_server(self):
        if self.runtime is None:
            return

        self.runtime.shutdown()
        self.runtime = None

    def _format_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def _postprocess_response(self, state: Any) -> dict:
        meta_info = state.get_meta_info("final_answer")
        result_dict = {
            "entire_response": state.text(),
            "response": state["final_answer"],
            "token_length": meta_info["prompt_tokens"] + meta_info["completion_tokens"],
        }
        return result_dict

    async def inference_thinkless_async(
        self,
        entry: dict,
        max_tokens: Optional[int] = 4096,
    ) -> dict:
        def _append_suffix(text: str) -> str:
            return text + "\n</think>\n\n"

        loop = asyncio.get_running_loop()

        # Cap max_tokens to max_total_tokens if set
        effective_max_tokens = max_tokens
        if self.max_total_tokens is not None and max_tokens is not None:
            effective_max_tokens = min(max_tokens, self.max_total_tokens)

        input_params = {
            "prompt": _append_suffix(self._format_prompt(entry)),
            "eos_token": self.eos_token,
            "max_tokens": effective_max_tokens,
            "sampling_params": self.sampling_params,
        }

        def _run_sync_thinkless():
            return run_rollout.run(stream=False, **input_params)

        state = await loop.run_in_executor(None, _run_sync_thinkless)
        return self._postprocess_response(state)

    async def inference_async(
        self,
        entry: dict,
        reasoning_tokens_budget: Optional[int] = 16384,
        answer_tokens_budget: Optional[int] = 4096,
        threshold: Optional[float] = None,
    ) -> dict:
        loop = asyncio.get_running_loop()

        # Cap total token budget to max_total_tokens if set
        effective_reasoning_budget = reasoning_tokens_budget
        effective_answer_budget = answer_tokens_budget
        if self.max_total_tokens is not None:
            total_requested = (reasoning_tokens_budget or 0) + (
                answer_tokens_budget or 0
            )
            if total_requested > self.max_total_tokens:
                scale_factor = self.max_total_tokens / total_requested
                effective_reasoning_budget = int(
                    (reasoning_tokens_budget or 0) * scale_factor
                )
                effective_answer_budget = (
                    self.max_total_tokens - effective_reasoning_budget
                )

        if not threshold:
            input_params = {
                "prompt": self._format_prompt(entry),
                "eos_token": self.eos_token,
                "max_tokens": effective_reasoning_budget + effective_answer_budget,
                "sampling_params": self.sampling_params,
            }

            def _run_sync_rollout():
                return run_rollout.run(stream=False, **input_params)

            state = await loop.run_in_executor(None, _run_sync_rollout)
            return self._postprocess_response(state)

        # Case 2: Threshold -> ThinkBrake generation
        input_params = {
            "prompt": self._format_prompt(entry),
            "threshold": threshold,
            "eot_token": self.eot_token,
            "eos_token": self.eos_token,
            "prefix_token": self.prefix_token,
            "suffix_token": self.suffix_token,
            "reasoning_tokens_budget": effective_reasoning_budget,
            "answer_tokens_budget": effective_answer_budget,
            "max_total_tokens": self.max_total_tokens,
            "sampling_params": self.sampling_params,
        }

        def _run_sync():
            return run_thinkbrake_step.run(stream=False, **input_params)

        state = await loop.run_in_executor(None, _run_sync)

        return self._postprocess_response(state)

    async def inference_oracle(self, entry: dict) -> dict:
        loop = asyncio.get_running_loop()

        input_params = {
            "prompt": self._format_prompt(entry),
            "eos_token": self.eos_token,
            "max_tokens": self.max_total_tokens,
            "sampling_params": self.sampling_params,
        }

        def _run_sync_rollout():
            return run_rollout.run(stream=False, **input_params)

        state = await loop.run_in_executor(None, _run_sync_rollout)
        return self._postprocess_response(state)


class DeepSeekHandler(BaseHandler):
    """Handler for DeepSeek R1/Distill models."""

    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path)
        self.eos_token = "<｜end of sentence｜>"
        self.prefix_token = ".\n"
        self.suffix_token = "\n\n"
        self.eot_token = "</think>"
        self.bot_token = "<think>"
        self.sampling_params = {
            "temperature": 0.6,
        }

    def _format_prompt(self, entry: dict) -> str:
        problem = entry["problem"]
        category = entry["category"]
        assistant = entry.get("assistant", None)

        formatted_prompt = f"<｜begin of sentence｜><｜User｜>{problem}\n\n"

        if category == "math":
            formatted_prompt += "Please reason step by step, and put your final answer within \\boxed{}."

        formatted_prompt += f"<｜Assistant｜><think>\n"

        if assistant:
            formatted_prompt += assistant

        return formatted_prompt


class Qwen3Handler(BaseHandler):
    """Handler for Qwen3 models."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
    ):
        super().__init__(pretrained_model_name_or_path)
        self.eos_token = "<|im_end|>"
        self.prefix_token = ".\n"
        self.suffix_token = "\n\n"
        self.eot_token = "</think>"
        self.bot_token = "<think>"
        self.sampling_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0,
        }

    def _format_prompt(self, entry: dict) -> str:
        problem = entry["problem"]
        category = entry["category"]
        assistant = entry.get("assistant", None)

        formatted_prompt = f"<|im_start|>user\n{problem}\n\n"

        if category == "math":
            formatted_prompt += "Please reason step by step, and put your final answer within \\boxed{}."
        elif category == "general":
            formatted_prompt += 'Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".'

        formatted_prompt += "<|im_end|>\n"
        formatted_prompt += f"<|im_start|>assistant\n<think>\n"

        if assistant:
            formatted_prompt += assistant

        return formatted_prompt


class Phi4Handler(BaseHandler):
    """Handler for Phi-4 models (Microsoft)."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
    ):
        super().__init__(pretrained_model_name_or_path)
        self.eos_token = "<|im_end|>"
        self.prefix_token = "."
        self.suffix_token = ""
        self.eot_token = "</think>"
        self.bot_token = "<think>"
        self.sampling_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 50,
        }

    def _format_prompt(self, entry: dict) -> str:
        problem = entry["problem"]

        system_prompt = (
            "You are Phi, a language model trained by Microsoft to help users. "
            "Your role as an assistant involves thoroughly exploring questions through a "
            "systematic thinking process before providing the final precise and accurate solutions. "
            "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, "
            "reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
            "Please structure your response into two main sections: Thought and Solution using the specified format: "
            "<think> {Thought section} </think> {Solution section}. "
            "In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations "
            "such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the "
            "current steps, refining any errors, and revisiting previous steps. "
            "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, "
            "systematically present the final solution that you deem correct. The Solution section should be logical, accurate, "
            "and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"
        )

        user_content = problem
        formatted_prompt = "<|im_start|>system<|im_sep|>" + system_prompt + "<|im_end|>"
        formatted_prompt += "<|im_start|>user<|im_sep|>" + user_content + "<|im_end|>"
        formatted_prompt += "<|im_start|>assistant<|im_sep|><think>"

        return formatted_prompt


# class GptOssHandler(BaseHandler):
#     """Handler for GPT-OSS (example) models."""

#     def __init__(self, pretrained_model_name_or_path: str):
#         super().__init__(pretrained_model_name_or_path)
#         self.eos_token = "<|return|>"
#         self.prefix_token = "."
#         self.suffix_token = ""
#         self.eot_token = "<|end|>"
#         self.sampling_params = {}

#         self.current_date = datetime.now().strftime("%Y-%m-%d")
#         self.reasoning_level = "high"

#     def _format_prompt(self, entry: dict) -> str:
#         problem = entry["problem"]

#         formatted_prompt = f"""<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
# Knowledge cutoff: 2024-06
# Current date: {self.current_date}

# Reasoning: {self.reasoning_level}

# # Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"""

#         formatted_prompt += f"<|start|>user<|message|>{problem}<|end|>"
#         formatted_prompt += "<|start|>assistant<|channel|>analysis<|message|>"

#         return formatted_prompt
