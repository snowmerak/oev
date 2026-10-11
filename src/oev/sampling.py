"""Request-local sampling policies shared by the optional inference runtimes."""

from collections import Counter
import math


class SampledToken(int):
    def __new__(cls, token, logprob, top):
        value = super().__new__(cls, token)
        value.logprob, value.top = logprob, top
        return value


def extended_sampling(options):
    return bool(options.seed is not None or options.top_k or options.min_p or options.presence_penalty
                or options.frequency_penalty or options.repetition_penalty != 1 or options.logit_bias or options.logprobs)


def sample_logits(logits, options, generated, rng, prompt=()):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not np.isfinite(logits).all():
        raise RuntimeError("model returned nonfinite generation logits")
    for token, bias in options.logit_bias.items():
        if token >= len(logits):
            raise ValueError(f"logit_bias token {token} is outside model vocabulary")
        logits[token] += bias
    for token, count in Counter(generated).items():
        logits[token] -= options.presence_penalty + options.frequency_penalty * count
    if options.repetition_penalty != 1:
        for token in set(prompt) | set(generated):
            logits[token] = logits[token] * options.repetition_penalty if logits[token] < 0 else logits[token] / options.repetition_penalty
    scaled = logits / options.temperature if options.temperature else logits
    if options.top_k and options.top_k < len(scaled):
        keep = np.argsort(-scaled, kind="stable")[:options.top_k]
        mask = np.ones(len(scaled), dtype=bool)
        mask[keep] = False
        scaled[mask] = -np.inf
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    if options.min_p:
        probabilities[probabilities < probabilities.max() * options.min_p] = 0
    if options.top_p < 1:
        order = np.argsort(-probabilities)
        remove = np.cumsum(probabilities[order] / probabilities.sum()) > options.top_p
        remove[1:] = remove[:-1].copy()
        remove[0] = False
        probabilities[order[remove]] = 0
    probabilities /= probabilities.sum()
    token = int(np.argmax(probabilities)) if options.temperature == 0 else int(rng.choice(len(probabilities), p=probabilities))
    if not options.logprobs:
        return token
    top = tuple((int(index), max(-9999.0, math.log(float(probabilities[index]))))
                for index in np.argsort(-probabilities)[:options.top_logprobs] if probabilities[index] > 0)
    return SampledToken(token, max(-9999.0, math.log(float(probabilities[token]))), top)
