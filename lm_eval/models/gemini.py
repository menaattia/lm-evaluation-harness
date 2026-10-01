import logging
import math
import os
import random
import re
import time
from functools import cached_property
from typing import Any, Dict, List

from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted

    print("Imported gemini.py!")
except ImportError as e:
    raise ImportError(
        "Gemini SDK not installed. Run `pip install google-generativeai`."
    )


eval_logger = logging.getLogger(__name__)


def _get_retry_delay(error: Exception, attempt: int) -> float:
    """Prefer Gemini's retry hint, falling back to exponential backoff."""
    match = re.search(r"Please retry in ([0-9.]+)s", str(error))
    if match:
        return math.ceil(float(match.group(1))) + random.uniform(0.5, 1.5)
    return min(2**attempt, 60) + random.uniform(0.0, 1.0)


def _generate_with_backoff(client, prompt, generation_config, max_retries=8):
    for attempt in range(max_retries + 1):
        try:
            return client.generate_content(
                prompt,
                generation_config=generation_config,
            )
        except ResourceExhausted as error:
            if attempt == max_retries:
                raise

            delay = _get_retry_delay(error, attempt)
            eval_logger.warning(
                "Gemini quota exhausted; retrying in %.1f seconds (%d/%d)",
                delay,
                attempt + 1,
                max_retries,
            )
            time.sleep(delay)


def _get_gemini_response_text(response):
    """Return all visible text while excluding Gemini thinking parts."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return ""

    return "".join(
        text
        for part in parts
        if not getattr(part, "thought", False)
        if (text := getattr(part, "text", None))
    )


@register_model("gemini")
class GeminiLM(LM):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 1024,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.kwargs = kwargs

        genai.configure(api_key=self.api_key)

        self.client = genai.GenerativeModel(model)

    @cached_property
    def api_key(self):
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY in your environment.")
        return key

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        if not requests:
            return []

        results = []
        for prompt, request_args in tqdm(
            [req.args for req in requests], disable=disable_tqdm
        ):
            stop = request_args.get("until", None)
            stop = handle_stop_sequences(stop, None) or []
            generation_config: Dict[str, Any] = {
                "max_output_tokens": request_args.get(
                    "max_gen_toks", self.max_tokens
                ),
                "stop_sequences": stop[:4],
            }
            if not self.model.startswith("gemini-3"):
                generation_config.update(
                    {
                        "temperature": request_args.get(
                            "temperature", self.temperature
                        ),
                        "top_p": self.top_p,
                        "top_k": self.top_k,
                    }
                )

            response = _generate_with_backoff(
                self.client,
                prompt,
                generation_config,
            )
            text = _get_gemini_response_text(response).strip()
            results.append(text)
            self.cache_hook.add_partial("generate_until", (prompt, request_args), text)
        return results

    def _model_call(self, inps):
        raise NotImplementedError("Gemini native API does not support logits.")

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError("Not needed.")

    def tok_encode(self, string: str) -> List[int]:
        return [string]

    def tok_decode(self, tokens: List[int]) -> str:
        return tokens[0]

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Gemini API does not return logprobs.")

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("Gemini API does not return logprobs.")
