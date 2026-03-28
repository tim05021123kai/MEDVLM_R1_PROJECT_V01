"""
LRU cache for inference results and analysis history tracking.
Avoids redundant inference calls for the same image+prompt combinations.
"""

import hashlib
import io
import time
from collections import OrderedDict
from PIL import Image


class InferenceCache:
    """LRU cache for model inference results keyed by (image_hash, prompt)."""

    def __init__(self, max_size=50):
        self._cache = OrderedDict()
        self._max_size = max_size

    def _hash_image(self, image):
        """Create a compact hash of a PIL Image."""
        buf = io.BytesIO()
        thumb = image.copy()
        thumb.thumbnail((128, 128))
        thumb.save(buf, format="PNG")
        return hashlib.md5(buf.getvalue()).hexdigest()

    def _make_key(self, image, prompt, backend=""):
        img_hash = self._hash_image(image)
        prompt_hash = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:12]
        return f"{backend}:{img_hash}:{prompt_hash}"

    def get(self, image, prompt, backend=""):
        key = self._make_key(image, prompt, backend)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]["result"]
        return None

    def put(self, image, prompt, result, backend=""):
        key = self._make_key(image, prompt, backend)
        self._cache[key] = {
            "result": result,
            "timestamp": time.time(),
        }
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    @property
    def size(self):
        return len(self._cache)


class AnalysisHistory:
    """Keeps a chronological history of all analyses performed in a session."""

    def __init__(self, max_entries=100):
        self._history = []
        self._max_entries = max_entries

    def add_entry(self, image_name, prompt, result, backend, confidence=None):
        entry = {
            "timestamp": time.time(),
            "image_name": image_name or "uploaded_image",
            "prompt": prompt[:200],
            "result": result,
            "backend": backend,
            "confidence": confidence,
        }
        self._history.append(entry)
        if len(self._history) > self._max_entries:
            self._history.pop(0)

    def get_history(self):
        return list(self._history)

    def format_history_display(self):
        if not self._history:
            return "No analysis history yet."

        lines = ["Analysis History", "=" * 60]
        for i, entry in enumerate(reversed(self._history), 1):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
            conf = f" | Confidence: {entry['confidence']:.0%}" if entry["confidence"] is not None else ""
            lines.append(f"\n#{i} [{ts}] Backend: {entry['backend']}{conf}")
            lines.append(f"  Prompt: {entry['prompt'][:100]}...")
            lines.append(f"  Result: {entry['result'][:150]}...")
        return "\n".join(lines)

    def clear(self):
        self._history.clear()

    @property
    def count(self):
        return len(self._history)
