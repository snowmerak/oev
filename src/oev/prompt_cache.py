"""Bounded, model-local prefix snapshots and request-local usage accounting."""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import sys
import re


def parse_cache_size(value: int | str) -> int:
    """Convert bytes or an explicit SI/IEC size to a whole number of bytes."""
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and len(value) <= 128:
        match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(B|[KMGT]I?B)?\s*", value, re.IGNORECASE)
        if match:
            amount, unit = match.groups()
            unit = (unit or "B").upper()
            exponent = "KMGT".index(unit[0]) + 1 if unit != "B" else 0
            multiplier = (1024 if "I" in unit else 1000) ** exponent
            whole, _, fraction = amount.partition(".")
            count, remainder = divmod(int(whole + fraction) * multiplier, 10 ** len(fraction))
            if not remainder:
                return count
    raise ValueError("prompt_cache_max_size must be nonnegative whole bytes or a size such as 256MB, 256MiB, 1GB, or 1GiB")


@dataclass
class CacheUsage:
    cached_tokens: int = 0


_usage = ContextVar("oev_prompt_cache_usage", default=None)


@contextmanager
def track_cache(usage):
    token = _usage.set(usage)
    try:
        yield
    finally:
        _usage.reset(token)


def record_hit(tokens):
    usage = _usage.get()
    if usage is not None:
        usage.cached_tokens = tokens


def tracked_tokens(source, usage):
    """Do not carry a ContextVar scope across an outward generator yield."""
    iterator = iter(source)
    while True:
        with track_cache(usage):
            try:
                token = next(iterator)
            except StopIteration:
                return
        yield token


def state_bytes(value):
    """Conservative size including containers and unique tensor allocations."""
    seen, allocations = set(), set()

    def visit(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        size = sys.getsizeof(item)
        if hasattr(item, "untyped_storage"):
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr())
            if key not in allocations:
                allocations.add(key)
                size += storage.nbytes()
        elif isinstance(item, dict):
            size += sum(visit(k) + visit(v) for k, v in item.items())
        elif isinstance(item, (list, tuple)):
            size += sum(visit(v) for v in item)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            size += visit(vars(item))
        elif hasattr(item, "nbytes"):
            # Visit cache objects' fields above, so their metadata is included
            # without counting an aggregate nbytes property a second time.
            # NumPy can already include its allocation in getsizeof; counting
            # it again is conservative. MLX's getsizeof excludes array data.
            size += int(item.nbytes)
        return size

    return visit(value)


class PrefixCache:
    """Callers hold their backend lock and clone snapshots before mutation.

    Only complete stored boundaries are reusable, including for sliding-window
    caches. Never crop a later snapshot to an arbitrary common-prefix length.
    The budget covers retained snapshots, not weights or active request state.
    """

    chunk_tokens = 256

    def __init__(self, max_bytes=0):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("prompt_cache_max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self._entries = OrderedDict()

    def lookup(self, tokens, *, readout_tokens=1):
        best = None
        for prefix in self._entries:
            # Leave at least one token to regenerate the final readout.
            if len(prefix) <= len(tokens) - readout_tokens and tuple(tokens[:len(prefix)]) == prefix:
                if best is None or len(prefix) > len(best):
                    best = prefix
        if best is None:
            return 0, None
        snapshot, _ = self._entries[best]
        self._entries.move_to_end(best)
        record_hit(len(best))
        return len(best), snapshot

    def store(self, tokens, state, nbytes, clone):
        if not self.max_bytes:
            return
        prefix = tuple(tokens)
        size = nbytes + state_bytes(prefix) + 256  # entry/LRU bookkeeping
        if size > self.max_bytes:
            return
        if prefix in self._entries:
            return  # immutable snapshot already exists
        while self.size_bytes + size > self.max_bytes:
            _, (_, removed) = self._entries.popitem(last=False)
            self.size_bytes -= removed
        snapshot = clone(state)
        self._entries[prefix] = snapshot, size
        self.size_bytes += size

    def clear(self):
        self._entries.clear()
        self.size_bytes = 0
