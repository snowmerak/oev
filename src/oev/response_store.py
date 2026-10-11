"""Bounded, process-local response history. Restarting the server clears it."""

from collections import OrderedDict
import json
from threading import Lock
import time


class ResponseStore:
    def __init__(self, *, max_entries=256, max_bytes=64 * 1024**2, ttl=3600):
        self.max_entries, self.max_bytes, self.ttl = max_entries, max_bytes, ttl
        self.entries, self.bytes, self.lock = OrderedDict(), 0, Lock()

    def _expire(self):
        now = time.monotonic()
        for key in list(self.entries):
            if self.entries[key][0] <= now:
                self._remove(key)

    def _remove(self, key):
        record = self.entries.pop(key)
        self.bytes -= record[1]
        return record

    def put(self, body, history):
        value = json.dumps((body, history), ensure_ascii=False, allow_nan=False).encode("utf-8")
        size = len(value)
        if size > self.max_bytes:
            raise ValueError("response history exceeds the in-memory storage limit")
        with self.lock:
            self._expire()
            if body["id"] in self.entries:
                self._remove(body["id"])
            while self.entries and (len(self.entries) >= self.max_entries or self.bytes + size > self.max_bytes):
                self._remove(next(iter(self.entries)))
            self.entries[body["id"]] = (time.monotonic() + self.ttl, size, value)
            self.bytes += size

    def get(self, key):
        with self.lock:
            self._expire()
            if key not in self.entries:
                raise KeyError(key)
            self.entries.move_to_end(key)
            return json.loads(self.entries[key][2])

    def delete(self, key):
        with self.lock:
            self._expire()
            if key not in self.entries:
                raise KeyError(key)
            self._remove(key)
