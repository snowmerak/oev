"""Local GGUF inference through an optional llama.cpp runtime."""

from __future__ import annotations

from pathlib import Path
from threading import Lock


class GGUFTokenizer:
    def __init__(self, llama, formatter):
        self.llama = llama
        self.formatter = formatter

    def apply_chat_template(self, messages, **kwargs):
        # The embedded Jinja template owns BOS and assistant-prefix insertion.
        return self.formatter(messages=messages, enable_thinking=False).prompt

    def encode(self, text, add_special_tokens=False):
        return self.llama.tokenize(text.encode("utf-8"), add_bos=add_special_tokens, special=True)

    def decode(self, token_ids, **kwargs):
        return self.llama.detokenize(
            token_ids, special=not kwargs.get("skip_special_tokens", False),
        ).decode("utf-8", errors="replace")


class GGUFBackend:
    """Reuse loaded weights and serialize requests around one native KV context."""

    def __init__(self, llama, tokenizer, model_name, api):
        self.llama = llama
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.context_window = llama.n_ctx()
        self._api = api
        self._vocab = api.llama_model_get_vocab(llama.model)
        self._lock = Lock()

    @classmethod
    def load(cls, source, *, device="auto", context_length=8192, gpu_layers=None, chat_template=None):
        path = Path(source).expanduser()
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise ValueError("GGUF models require the path to an existing .gguf file")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("GGUF device must be auto, cpu, cuda, or mps; hybrid-cuda is HF-only")
        if type(context_length) is not int or context_length < 1:
            raise ValueError("context_length must be a positive integer")
        if gpu_layers is not None and (type(gpu_layers) is not int or gpu_layers < -1):
            raise ValueError("gpu_layers must be -1 or a nonnegative integer")
        layers = gpu_layers if gpu_layers is not None else -1 if device in {"cuda", "mps"} else 0
        if device == "cpu" and layers != 0:
            raise ValueError("device=cpu requires gpu_layers=0")
        if device in {"cuda", "mps"} and layers == 0:
            raise ValueError("a GPU device requires nonzero gpu_layers")
        try:
            from llama_cpp import Llama, llama_cpp as api
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        except ImportError as exc:
            raise RuntimeError("Install GGUF support with: uv sync --extra gguf --extra serve") from exc
        if layers != 0 and not api.llama_supports_gpu_offload():
            raise ValueError("this llama-cpp-python build does not support GPU offload")
        llama = Llama(model_path=str(path.resolve()), n_ctx=context_length, n_gpu_layers=layers,
                      logits_all=False, verbose=False)
        try:
            template = Path(chat_template).expanduser().read_text(encoding="utf-8") if chat_template else llama.metadata.get("tokenizer.chat_template")
            if not isinstance(template, str) or not template.strip():
                raise ValueError("GGUF has no embedded chat template; provide --chat-template with a Jinja template file")

            def special_text(token):
                return llama.detokenize([token], special=True).decode("utf-8") if token >= 0 else ""

            formatter = Jinja2ChatFormatter(
                template=template, bos_token=special_text(llama.token_bos()),
                eos_token=special_text(llama.token_eos()), add_generation_prompt=True,
            )
            return cls(llama, GGUFTokenizer(llama, formatter), str(path.resolve()), api)
        except Exception:
            llama.close()
            raise

    def is_end_token(self, token):
        # Includes end-of-turn markers as well as the primary EOS token.
        return self._api.llama_vocab_is_eog(self._vocab, token)

    def _validate_context(self, input_ids, output_tokens=0):
        if not input_ids or len(input_ids) + output_tokens > self.context_window:
            raise ValueError(f"prompt plus max output tokens exceeds GGUF context limit {self.context_window}")

    def _logits(self):
        # Read the final row directly: logits_all=True stores every prompt row.
        return self._api.llama_get_logits_ith(self.llama.ctx, -1)

    def selected_logits(self, input_ids, answer_token_ids):
        self._validate_context(input_ids)
        with self._lock:
            try:
                self.llama.reset()
                self.llama.eval(input_ids)
                logits = self._logits()
                if not logits:
                    raise RuntimeError("llama.cpp returned no logits")
                return [float(logits[token]) for token in answer_token_ids]
            finally:
                self.llama.reset()

    def generate_tokens(self, input_ids, options, cancel):
        import numpy as np

        self._validate_context(input_ids, options.max_new_tokens)
        while not cancel.is_set():
            if self._lock.acquire(timeout=0.1):
                break
        else:
            return
        try:
            self.llama.reset()
            # Bound cancellation latency to one native prompt batch/forward.
            for start in range(0, len(input_ids), self.llama.n_batch):
                if cancel.is_set():
                    return
                self.llama.eval(input_ids[start:start + self.llama.n_batch])
            rng = np.random.default_rng()
            for index in range(options.max_new_tokens):
                if cancel.is_set():
                    return
                pointer = self._logits()
                if not pointer:
                    raise RuntimeError("llama.cpp returned no logits")
                logits = np.ctypeslib.as_array(pointer, shape=(self.llama.n_vocab(),)).astype(np.float64)
                if not np.isfinite(logits).all():
                    raise RuntimeError("model returned nonfinite generation logits")
                if options.temperature == 0:
                    token = int(logits.argmax())
                else:
                    # Match the HF temperature/top-p policy, without implicit penalties.
                    logits = (logits - logits.max()) / options.temperature
                    probabilities = np.exp(logits)
                    probabilities /= probabilities.sum()
                    if options.top_p < 1:
                        order = np.argsort(-probabilities)
                        remove = probabilities[order].cumsum() > options.top_p
                        remove[1:] = remove[:-1].copy()
                        remove[0] = False
                        probabilities[order[remove]] = 0
                        probabilities /= probabilities.sum()
                    token = int(rng.choice(len(probabilities), p=probabilities))
                yield token
                if self.is_end_token(token) or cancel.is_set():
                    return
                if index + 1 < options.max_new_tokens:
                    self.llama.eval([token])
        finally:
            try:
                self.llama.reset()
            finally:
                self._lock.release()
