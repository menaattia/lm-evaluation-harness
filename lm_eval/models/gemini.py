import os
from typing import List, Tuple, Dict, Union

from tqdm import tqdm
from functools import cached_property

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import handle_stop_sequences

try:
    import google.generativeai as genai
    print("Imported gemini.py!")
except ImportError as e:
    raise ImportError(
        "Gemini SDK not installed. Run `pip install google-generativeai`."
    )

def _get_gemini_response_text(response):
    # Safely get response text from Gemini API response object
    if hasattr(response, "candidates") and response.candidates:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            parts = candidate.content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    return ""

@register_model("gemini")
class GeminiLM(LM):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        max_tokens: int = 128,
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
        for prompt, request_args in tqdm([req.args for req in requests], disable=disable_tqdm):
            stop = request_args.get("until", None)
            stop = handle_stop_sequences(stop, None) or []
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": request_args.get("max_gen_toks", self.max_tokens),
                    "temperature": request_args.get("temperature", self.temperature),
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "stop_sequences": stop[:4],
                }
            )
            text = _get_gemini_response_text(response)
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