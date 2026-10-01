import os
import re
from functools import cached_property
from typing import Any, Dict, List

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences

try:
    import anthropic
    print("Imported anthropic.py!")
except ImportError as e:
    raise ImportError(
        "Anthropic SDK not installed. Run `pip install anthropic`."
    )


def _supports_sampling_parameters(model: str) -> bool:
    """Return whether Anthropic accepts temperature/top_p for this model."""
    if model.startswith("claude-mythos"):
        return False

    match = re.match(r"^claude-(sonnet|opus|fable)-(\d+)(?:-(\d+))?", model)
    if not match:
        return True

    family, major, minor = match.groups()
    version = (int(major), int(minor or 0))
    return version < (5, 0) and not (family == "opus" and version >= (4, 7))


@register_model("claude")
class ClaudeLM(LM):
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs

        self.client = anthropic.Anthropic(api_key=self.api_key)

    @cached_property
    def api_key(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("Set ANTHROPIC_API_KEY in your environment.")
        return key

    @property
    def tokenizer_name(self):
        """Return the tokenizer name for this model."""
        return self.model

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []

        results = []
        for prompt, request_args in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            stop = request_args.get("until", None)
            stop = handle_stop_sequences(stop, None) or []
            stop = [s for s in stop if s and s.strip()]
            # Claude expects stop_sequences as a list of strings (max 4 allowed).
            generation_args: Dict[str, Any] = {
                "model": self.model,
                "max_tokens": request_args.get("max_gen_toks", self.max_tokens),
                "stop_sequences": stop[:4],
                "messages": [{"role": "user", "content": prompt}],
            }
            if _supports_sampling_parameters(self.model):
                generation_args["temperature"] = request_args.get(
                    "temperature", self.temperature
                )
                generation_args["top_p"] = request_args.get("top_p", self.top_p)

            response = self.client.messages.create(**generation_args)
            text_blocks = [
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ]
            if not text_blocks:
                raise ValueError(
                    "Claude returned no text block: "
                    f"stop_reason={response.stop_reason}, "
                    f"content={response.content!r}"
                )
            text = "".join(text_blocks)
            results.append(text)
            self.cache_hook.add_partial("generate_until", (prompt, request_args), text)
        return results

    def _model_call(self, inps):
        raise NotImplementedError("Claude native API does not support logits.")

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError("Not needed.")

    def tok_encode(self, string: str) -> List[int]:
        raise NotImplementedError("Not needed.")

    def tok_decode(self, tokens: List[int]) -> str:
        raise NotImplementedError("Not needed.")

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude API does not return logprobs.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Claude API does not return logprobs.")
