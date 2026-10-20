"""Loaded by the venv's Python at start (site-packages/sitecustomize.py, placed by `python3 -m mac venv`).

vllm-mlx batches requests with mlx-lm's cache classes and hands each request's recurrent layers back as mlx-lm's
ArraysCache, while mlx-vlm's Qwen3.5 DeltaNet layers call `update_window(..., lengths=cache.lengths)` and
`update_recurrent` on them, which only mlx-vlm's ArraysCache has. Without this, a prompt that resumes from the prefix cache fails with
"'ArraysCache' object has no attribute 'update_window'". This gives mlx-lm's class the same two updates (without mlx-vlm's speculative history)."""

try:
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache as _LmArraysCache
except Exception:  # not this venv's concern
    _LmArraysCache = None

if _LmArraysCache is not None and not hasattr(_LmArraysCache, "update_window"):

    def _update_window(self, index, source, width, *, lengths=None):
        """Keep the trailing causal window of `source` (the last `width` positions, or each row's at its length)."""
        width = int(width)
        length = source.shape[1] - width
        if width < 0 or length < 0:
            raise ValueError("Invalid causal cache window width.")
        if lengths is None:
            state = mx.contiguous(source[:, length : length + width])
        else:
            positions = mx.clip(lengths, 0, length)[:, None] + mx.arange(width)
            positions = positions.reshape(*positions.shape, *([1] * (source.ndim - 2)))
            state = mx.take_along_axis(source, positions, axis=1)
        self.cache[index] = state
        return state

    def _update_recurrent(self, index, length, update):
        """Run the layer's state transition on the kept state (`update(state, None)` gives output and the new state)."""
        if length < 1:
            raise ValueError("Recurrent updates must contain at least one token.")
        output, state = update(self.cache[index], None)[:2]
        self.cache[index] = state
        return output, state

    _LmArraysCache.update_window = _update_window
    _LmArraysCache.update_recurrent = _update_recurrent
    if not isinstance(getattr(_LmArraysCache, "lengths", None), property):
        _LmArraysCache.lengths = property(lambda self: self.__dict__.get("lengths"), lambda self, v: self.__dict__.__setitem__("lengths", v))
