"""The head on MLX, for Apple Silicon.

Nothing about the model is reimplemented here.  mlx-lm already carries this architecture -- 24
layers, eighteen of them gated-delta linear attention and six full attention -- so this file is
only the typed head: run the backbone, read the hidden state at each answer slot, score it
against the option label tokens.

Two details about the checkpoint are worth stating, because both look like bugs otherwise.

The config calls itself ``qwen3_5_text`` and mlx-lm resolves a model type to a module name, so
the import would fail on a name that no module has.  The architecture it denotes is the one
``mlx_lm.models.qwen3_5`` implements, and ``load(model_config=...)`` is the documented way to say
so.  It is a rename, not a reinterpretation.

The option logits come from ``embed_tokens.as_linear`` rather than from gathering 255 rows of the
embedding matrix.  That is the same tied-weight projection the model's own head applies, and it
costs about 0.5 GFLOP against the ~0.8 TFLOP a couple of hundred tokens of prefill already cost,
so the arithmetic is free at this scale.  What it buys is that a quantised checkpoint works
unchanged: ``as_linear`` knows how to unpack a ``QuantizedEmbedding`` and row indexing does not.
"""
from __future__ import annotations

import numpy as np

from ..types import MAX_OPTIONS

# The mlx-lm module that implements the architecture the checkpoint calls "qwen3_5_text".
MLX_MODEL_TYPE = "qwen3_5"

# float16 rather than the checkpoint's own bfloat16, and the difference is not cosmetic.  Both
# types cost two bytes; bfloat16 spends them on exponent range and keeps 8 mantissa bits where
# float16 keeps 11.  Nothing in a 1.9B forward pass needs that range, and the missing three bits
# land directly on the answer: measured against float32 on the CPU over the recorded items,
# bfloat16 moves a probability by up to 2.4e-2 where float16 moves it by 7.1e-3.  A model whose
# output is a calibrated probability should not spend three times its error budget on a type it
# has no use for.  `scripts/compare_backends.py` produced those two numbers; re-run it before
# changing this line.
DEFAULT_DTYPE = "float16"


def _backbone(model):
    """The layer stack that returns hidden states, before the tied projection to the vocabulary.

    mlx-lm wraps this architecture as Model -> language_model -> model, and has moved the
    convenience accessors around between releases, so the nesting is walked rather than named.
    """
    inner = getattr(model, "language_model", model)
    return getattr(inner, "model", inner)


def mlx_available() -> bool:
    """True if this machine can run the MLX backend at all."""
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


class MLXBackend:
    name = "mlx"

    def __init__(self, model, tokenizer):
        import mlx.core as mx

        from ..prompt import option_label_ids
        self.mx = mx
        self.model = model
        self.backbone = _backbone(model)
        self.tokenizer = tokenizer
        self.device = f"mlx ({_dtype_name(self.backbone)})"
        self._labels = mx.array(option_label_ids(tokenizer))
        pad = tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else (tokenizer.eos_token_id or 0)

    @classmethod
    def load(cls, name_or_path: str, *, dtype: str | None = DEFAULT_DTYPE,
             **kw) -> MLXBackend:
        """Load straight from the published checkpoint; no separate MLX repo is needed.

        The weights are a single bf16 safetensors file, the embeddings are tied, and mlx-lm's
        ``sanitize`` already drops the multi-token-prediction head the checkpoint ships, so
        there is nothing for a conversion step to do.  A directory produced by
        ``mlx_lm.convert`` -- a quantised one, say -- loads through this same path.
        """
        from mlx_lm import load
        from transformers import AutoTokenizer

        # The tokenizer is loaded from transformers rather than taken from mlx-lm's wrapper, so
        # that both backends tokenise through byte-identical code and a parity failure can only
        # be the numerics.
        tok = AutoTokenizer.from_pretrained(name_or_path)
        model, _ = load(name_or_path, model_config={"model_type": MLX_MODEL_TYPE}, **kw)
        if dtype is not None:
            _cast(model, dtype)
        return cls(model, tok)

    def slot_logits(self, ids, attn, slot_idx, slot_batch, n_options) -> np.ndarray:
        """`attn` is accepted and ignored; see the note on right padding below."""
        mx = self.mx
        backbone = self.backbone
        # No attention mask is passed, and none is needed.  The padding is on the right, and
        # every layer here is causal: a full-attention layer masks position i to j <= i, the
        # gated-delta recurrence at i has only accumulated j <= i, and the depthwise conv1d is
        # left-padded.  A real token therefore cannot see a pad token, and right alignment keeps
        # every real token's rotary position identical to its position when run alone.  The
        # `test_padding_does_not_change_the_answer` case pins this down rather than trusting it.
        hidden = backbone(mx.array(np.asarray(ids, dtype=np.int32)))            # [B, T, H]
        h = hidden[mx.array(np.asarray(slot_batch, dtype=np.int32)),
                   mx.array(np.asarray(slot_idx, dtype=np.int32))]              # [N, H]
        logits = backbone.embed_tokens.as_linear(h)[:, self._labels]            # [N, MAX_OPTIONS]
        logits = logits.astype(mx.float32)
        mx.eval(logits)
        out = np.array(logits, dtype=np.float32)
        past = np.arange(MAX_OPTIONS)[None, :] >= np.asarray(n_options)[:, None]
        return np.where(past, -np.inf, out).astype(np.float32)

    def synchronize(self) -> None:
        self.mx.synchronize()


def _dtype_name(backbone) -> str:
    """How the weights actually landed, for `decider.device` to report: "bfloat16", "4bit", ..."""
    emb = backbone.embed_tokens
    bits = getattr(emb, "bits", None)
    if bits is not None:
        return f"{bits}bit"
    return str(emb.weight.dtype).rsplit(".", 1)[-1]


def _cast(model, dtype) -> None:
    """Cast the floating parameters, honouring the model's own exclusions.

    The gated-delta recurrence keeps `A_log` in float32 -- mlx-lm's `cast_predicate` says so, and
    the checkpoint agrees with its `mamba_ssm_dtype: float32` -- so this walks parameter paths
    rather than casting the tree flat, exactly as `mlx_lm.convert` does.

    A quantised checkpoint is left alone: its weights are packed integers that a float cast would
    either skip or, for the scales, silently degrade, and the dtype it was quantised at is not a
    runtime choice any more.
    """
    import mlx.core as mx
    from mlx.utils import tree_map_with_path

    if getattr(_backbone(model).embed_tokens, "bits", None) is not None:
        return
    dtype = getattr(mx, dtype) if isinstance(dtype, str) else dtype
    keep = getattr(model, "cast_predicate", lambda _: True)

    def one(path, v):
        return v.astype(dtype) if keep(path) and mx.issubdtype(v.dtype, mx.floating) else v

    model.update(tree_map_with_path(one, model.parameters()))
