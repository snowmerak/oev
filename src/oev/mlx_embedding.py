"""Native MLX-VLM EmbeddingGemma2 encoder, including vision and audio towers."""

from threading import Lock

from .embedding_inputs import configure_audio_processor, decode_inputs


class MLXMultimodalEmbeddingBackend:
    def __init__(self, model, processor, config, prompts, mx, stream):
        self.model, self.processor, self._mx, self._stream = model, processor, mx, stream
        configure_audio_processor(processor)
        self.dimension = config["text_config"]["embedding_dim"]
        self.context_window = min(config["text_config"].get("max_position_embeddings", 8192), 8192)
        self.modalities = frozenset({"text"} | ({"image"} if config.get("vision_config") else set())
                                    | ({"audio"} if config.get("audio_config") else set()))
        self.prompts = prompts
        self._lock = Lock()

    @classmethod
    def load(cls, path, config, prompts, mx, stream):
        try:
            from mlx_vlm.embedding_loader import load_embedding_model
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install MLX multimodal embeddings with: uv sync --extra mlx-embeddings --extra serve") from exc
        with mx.stream(stream):
            # Eager loading and all subsequent evaluation use the same shared stream.
            model = load_embedding_model(path, lazy=False)
            processor = AutoProcessor.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
            mx.synchronize(stream)
        return cls(model, processor, config, prompts, mx, stream)

    def embed(self, texts, max_input_tokens):
        return self._embed([{"text": text} for text in texts], max_input_tokens)

    def embed_multimodal(self, inputs, max_input_tokens, *, prompt=""):
        return self._embed(decode_inputs(inputs), max_input_tokens, prompt=prompt)

    def _embed(self, inputs, max_input_tokens, *, prompt=""):
        from .embedding_backends import _check_tokens

        with self._lock:
            prepared, counts = [], []
            for item in inputs:
                content = []
                for key, value in item.items():
                    values = value if isinstance(value, list) else [value]
                    for part in values:
                        content.append({"type": key, key: part["array"] if key == "audio" else part})
                messages = ([{"role": "system", "content": prompt}] if prompt else [])
                messages.append({"role": "user", "content": content})
                tensors = self.processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="np",
                    processor_kwargs={"text_kwargs": {"padding": True, "truncation": False},
                                      "audio_kwargs": {"sampling_rate": 16000, "truncation": False}},
                )
                counts.append(int(tensors["attention_mask"].sum()))
                prepared.append(tensors)
            _check_tokens(counts, min(max_input_tokens, self.context_window))
            mx = self._mx
            vectors = []
            try:
                with mx.stream(self._stream):
                    for tensors in prepared:
                        output = self.model(**{key: mx.array(value) for key, value in tensors.items()})
                        embedding = output.text_embeds.astype(mx.float32)
                        mx.eval(embedding)
                        vectors.append(embedding.tolist()[0])
                return vectors, sum(counts)
            finally:
                mx.synchronize(self._stream)
