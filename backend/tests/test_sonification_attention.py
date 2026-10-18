"""Tests for Phase 5.8a attention-aware sonification telemetry.

Validates that:
1. The production extraction helper `_extract_attention_focus` accurately computes
   per-layer attention focus for known peaked/uniform tensors and rounds to 4 decimals.
2. Transformer-layer indexing is preserved when layers are missing or malformed (emits None).
3. Malformed and unavailable attentions cleanly fall back to empty list or None slots.
4. Prompt token length is validated against MAX_TOKENS in `generate_steps`.
5. Prefill steps omit full attention matrices, while ordinary decode steps extract
   bounded attention focus under MAX_ATTENTION_CONTEXT.
6. Models without attention support and GGUF backends honestly report empty attention_focus: [].
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import (
    MAX_TOKENS,
    ActivationEngine,
    ModelEngine,
    TokenizedTooLong,
    _extract_attention_focus,
)


def test_extract_attention_focus_known_tensors_and_math():
    """Verify production _extract_attention_focus against known peaked, uniform, and multi-query tensors."""
    num_heads = 4
    kv_len = 10

    # Peaked attention: each head focuses 100% on token 0
    peaked = torch.zeros(1, num_heads, 1, kv_len)
    peaked[:, :, 0, 0] = 1.0
    res_peaked = _extract_attention_focus((peaked,))
    assert res_peaked == [1.0]

    # Uniform attention: each head spreads 1/10 across all tokens
    uniform = torch.full((1, num_heads, 1, kv_len), 0.1)
    res_uniform = _extract_attention_focus((uniform,))
    assert res_uniform == [0.1]

    # Multiple layers together
    res_multi = _extract_attention_focus((peaked, uniform))
    assert res_multi == [1.0, 0.1]

    # Verify query position indexing: helper must inspect the LAST query position (q_len - 1)
    # Shape: (1, num_heads=2, q_len=3, kv_len=4)
    multi_q = torch.zeros(1, 2, 3, 4)
    multi_q[:, :, 0, :] = 0.9  # pos 0
    multi_q[:, :, 1, :] = 0.5  # pos 1
    multi_q[:, :, 2, :] = 0.25  # pos 2 (last position)
    res_q = _extract_attention_focus((multi_q,))
    assert res_q == [0.25]


def test_extract_attention_focus_rounding():
    """Verify that head average attention focus is rounded to 4 decimal places."""
    # 2 heads with peak weights 0.12345 and 0.12347 -> mean = 0.12346 -> rounded = 0.1235
    t = torch.zeros(1, 2, 1, 5)
    t[0, 0, 0, 0] = 0.12345
    t[0, 1, 0, 0] = 0.12347
    res = _extract_attention_focus((t,))
    assert res == [0.1235]


def test_extract_attention_focus_layer_index_preservation_with_missing_layer():
    """Verify that missing/None layers preserve their slot as None and never shift later layer indices."""
    num_heads = 2
    kv_len = 5
    layer0 = torch.zeros(1, num_heads, 1, kv_len)
    layer0[:, :, 0, 0] = 0.85
    layer2 = torch.zeros(1, num_heads, 1, kv_len)
    layer2[:, :, 0, 0] = 0.40

    # Layer 1 is None in the middle
    res = _extract_attention_focus((layer0, None, layer2))
    assert len(res) == 3
    assert res[0] == 0.85
    assert res[1] is None  # Slot 1 is preserved, not dropped
    assert res[2] == 0.40  # Layer 2 remains at index 2

    # Layer 0 is None at start
    res_start = _extract_attention_focus((None, layer2))
    assert len(res_start) == 2
    assert res_start[0] is None
    assert res_start[1] == 0.40

    # Layer 1 is None at end
    res_end = _extract_attention_focus((layer0, None))
    assert len(res_end) == 2
    assert res_end[0] == 0.85
    assert res_end[1] is None


def test_extract_attention_focus_malformed_and_unavailable():
    """Verify graceful handling of unavailable, empty, or malformed attention structures."""
    # Unavailable / empty
    assert _extract_attention_focus(None) == []
    assert _extract_attention_focus(()) == []
    assert _extract_attention_focus([]) == []

    # Valid layer for comparison
    valid = torch.zeros(1, 2, 1, 4)
    valid[:, :, 0, 0] = 0.75

    # 3D tensor (missing batch or head dimension)
    malformed_3d = torch.zeros(2, 1, 4)
    # 2D tensor
    malformed_2d = torch.zeros(1, 4)
    # Empty dimension
    empty_dim = torch.zeros(1, 0, 1, 4)
    # Non-tensor object
    non_tensor = "not_a_tensor"  # type: ignore
    # Tensor with NaNs
    nan_tensor = torch.full((1, 2, 1, 4), float("nan"))
    # Tensor with Inf
    inf_tensor = torch.full((1, 2, 1, 4), float("inf"))

    attentions = (valid, malformed_3d, malformed_2d, empty_dim, non_tensor, nan_tensor, inf_tensor, valid)
    res = _extract_attention_focus(attentions)

    assert len(res) == 8
    assert res[0] == 0.75
    assert res[1] is None
    assert res[2] is None
    assert res[3] is None
    assert res[4] is None
    assert res[5] is None
    assert res[6] is None
    assert res[7] == 0.75


@pytest.fixture
def create_mock_service():
    """Factory fixture to create an initialized ModelEngine test instance with mock tokenizer."""
    def _create(
        model: nn.Module,
        model_id: str = "test-model",
        num_layers: int = 1,
        num_heads: int = 2,
        hidden_size: int = 16,
        tokenizer: MagicMock | None = None,
    ) -> ModelEngine:
        service = ModelEngine.__new__(ModelEngine)
        service.model = model
        service.device = "cpu"
        service.model_id = model_id
        service.mode = "causal_lm"
        service._lock = threading.Lock()
        service.num_layers = num_layers
        service.num_heads = num_heads
        service.hidden_size = hidden_size
        service.revision = "test-rev"
        service.activation_engine = ActivationEngine()
        service._layer_elapsed = {}
        service._layer_timing_lock = threading.Lock()
        service._catalog = None

        if tokenizer is None:
            tokenizer = MagicMock()
            tokenizer.chat_template = None
            tokenizer.side_effect = lambda prompt, **kw: {"input_ids": torch.tensor([[1, 2]])}
            tokenizer.encode.return_value = [1, 2]
            tokenizer.decode.side_effect = lambda ids, **kw: f"tok_{ids[0] if ids else ''}"
            tokenizer.eos_token_id = 9999
        service.tokenizer = tokenizer
        return service

    return _create


def test_generate_steps_enforces_max_tokens(create_mock_service):
    """generate_steps validates prompt length before emitting metadata or running forward passes."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(model_type="dummy")

        def forward(self, input_ids, **kwargs):
            return types.SimpleNamespace(logits=torch.zeros(1, 1, 10), past_key_values=None, hidden_states=())

    tokenizer = MagicMock()
    tokenizer.chat_template = None
    tokenizer.side_effect = lambda prompt, **kw: {
        "input_ids": torch.tensor([[1] * (MAX_TOKENS + 5)])
    }
    service = create_mock_service(
        model=DummyModel(),
        model_id="test-model",
        tokenizer=tokenizer,
    )

    with pytest.raises(TokenizedTooLong) as exc_info:
        list(service.generate_steps(prompt="excessively long prompt", max_new_tokens=2, use_chat_template=False))

    assert exc_info.value.n_tokens == MAX_TOKENS + 5
    assert exc_info.value.cap == MAX_TOKENS


def test_generate_steps_prefill_and_context_cap_attentions(create_mock_service):
    """Verify prefill omits attentions and decode requests attentions only under MAX_ATTENTION_CONTEXT."""
    captured_kwargs: list[dict] = []

    class AttnRecordingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                vocab_size=50,
                hidden_size=16,
                is_encoder_decoder=False,
                model_type="llama",
            )
            self.embed = nn.Embedding(50, 16)
            self.lm_head = nn.Linear(16, 50, bias=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True,
                    output_hidden_states=True, output_attentions=False):
            captured_kwargs.append({"output_attentions": output_attentions})
            x = self.embed(input_ids)
            logits = self.lm_head(x)
            seq_len = input_ids.shape[1]
            kv_len = seq_len if past_key_values is None else (past_key_values + seq_len)
            attn = torch.zeros(1, 2, seq_len, kv_len)
            attn[:, :, -1, 0] = 0.8
            attentions = (attn,) if output_attentions else None
            new_pkv = kv_len
            return types.SimpleNamespace(
                logits=logits,
                past_key_values=new_pkv,
                hidden_states=(x,),
                attentions=attentions,
            )

    model = AttnRecordingModel()
    service = create_mock_service(
        model=model,
        model_id="test-attn-recording",
        num_layers=1,
        num_heads=2,
        hidden_size=16,
    )

    # Step 0 (prefill) + Step 1 (decode)
    frames = list(
        service.generate_steps(
            prompt="test prompt",
            max_new_tokens=2,
            top_k=5,
            use_chat_template=False,
            include_catalog=False,
            decoding_mode="greedy",
        )
    )
    token_frames = [f for f in frames if f.get("type") == "token"]
    assert len(token_frames) == 2

    # Prefill step: output_attentions must be False and attention_focus empty
    assert captured_kwargs[0]["output_attentions"] is False
    assert token_frames[0]["phase"] == "prefill"
    assert token_frames[0]["attention_focus"] == []

    # Decode step: output_attentions must be True and attention_focus populated
    assert captured_kwargs[1]["output_attentions"] is True
    assert token_frames[1]["phase"] == "decode"
    assert token_frames[1]["attention_focus"] == [0.8]


def test_generate_steps_context_cap_disables_attentions(create_mock_service, monkeypatch):
    """Verify that when context exceeds MAX_ATTENTION_CONTEXT, attention extraction is omitted."""
    # Set cap to 1 token so that step 2 (cache_len_in=2) exceeds the cap
    monkeypatch.setattr("app.model.MAX_ATTENTION_CONTEXT", 1)

    captured_kwargs: list[dict] = []

    class CapCheckModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                vocab_size=50,
                hidden_size=16,
                is_encoder_decoder=False,
                model_type="llama",
            )
            self.embed = nn.Embedding(50, 16)
            self.lm_head = nn.Linear(16, 50, bias=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True,
                    output_hidden_states=True, output_attentions=False):
            captured_kwargs.append({"output_attentions": output_attentions})
            x = self.embed(input_ids)
            logits = self.lm_head(x)
            attn = (torch.zeros(1, 2, 1, 2),) if output_attentions else None
            return types.SimpleNamespace(
                logits=logits,
                past_key_values="dummy_pkv",
                hidden_states=(x,),
                attentions=attn,
            )

    model = CapCheckModel()
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda prompt, **kw: {"input_ids": torch.tensor([[1]])}
    tokenizer.encode.return_value = [1]
    tokenizer.decode.side_effect = lambda ids, **kw: f"tok_{ids[0] if ids else ''}"
    tokenizer.eos_token_id = 9999
    service = create_mock_service(
        model=model,
        model_id="test-cap-check",
        num_layers=1,
        num_heads=2,
        hidden_size=16,
        tokenizer=tokenizer,
    )

    gen = service.generate_steps(
        prompt="test prompt",
        max_new_tokens=3,
        top_k=5,
        use_chat_template=False,
        include_catalog=False,
        decoding_mode="greedy",
    )
    # Read meta
    next(gen)
    # Step 0 (prefill, positions_done=0 -> 1): prefill omits attentions
    tf0 = next(gen)
    assert tf0["phase"] == "prefill"
    assert captured_kwargs[0]["output_attentions"] is False
    assert tf0["attention_focus"] == []

    # Step 1 (decode, cache_len_in=1 <= MAX_ATTENTION_CONTEXT=1): attentions requested
    tf1 = next(gen)
    assert tf1["phase"] == "decode"
    assert captured_kwargs[1]["output_attentions"] is True
    assert len(tf1["attention_focus"]) == 1

    # Step 2 (decode, cache_len_in=2 > MAX_ATTENTION_CONTEXT=1): attentions omitted
    tf2 = next(gen)
    assert tf2["phase"] == "decode"
    assert captured_kwargs[2]["output_attentions"] is False
    assert tf2["attention_focus"] == []


def test_custom_model_without_output_attentions_generates_successfully(create_mock_service):
    """Verify custom/synthetic models without output_attentions support generate cleanly
    and emit honest fallback attention_focus: []."""
    class SyntheticNoAttnModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(
                vocab_size=50,
                hidden_size=16,
                is_encoder_decoder=False,
                model_type="llama",
            )
            self.embed = nn.Embedding(50, 16)
            self.lm_head = nn.Linear(16, 50, bias=False)

        def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=True):
            # Model forward signature explicitly does not accept **kwargs or output_attentions
            x = self.embed(input_ids)
            logits = self.lm_head(x)
            return types.SimpleNamespace(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=(x,),
            )

    model = SyntheticNoAttnModel()
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda prompt, **kw: {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.side_effect = lambda ids, **kw: f"tok_{ids[0] if ids else ''}"
    tokenizer.eos_token_id = 9999
    service = create_mock_service(
        model=model,
        model_id="test-no-attn",
        num_layers=1,
        num_heads=1,
        hidden_size=16,
        tokenizer=tokenizer,
    )

    frames = list(
        service.generate_steps(
            prompt="test prompt",
            max_new_tokens=2,
            top_k=5,
            use_chat_template=False,
            include_catalog=False,
            decoding_mode="greedy",
        )
    )
    token_frames = [f for f in frames if f.get("type") == "token"]
    assert len(token_frames) == 2
    for tf in token_frames:
        assert "attention_focus" in tf
        # Honest fallback: empty list when model does not support attention extraction
        assert tf["attention_focus"] == []


def test_gguf_frame_attention_focus_honesty():
    """GGUF backend yields empty attention_focus list (llama.cpp exposes no attention matrices)."""
    import numpy as np
    from app.gguf_engine import GGUFEngine

    class FakeLlama:
        def __init__(self):
            self.metadata = {"general.name": "fake-model"}
            self.scores = np.array([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])
            self.n_tokens = 1

        def reset(self):
            self.n_tokens = 1

        def tokenize(self, prompt, add_bos=True):
            return [0]

        def eval(self, token_ids):
            self.n_tokens += len(token_ids)

        def n_vocab(self):
            return 3

        def n_ctx(self):
            return 128

        def token_eos(self):
            return 2

        def detokenize(self, token_ids):
            tid = int(token_ids[0])
            return {0: b"prompt", 1: b"hello", 2: b""}.get(tid, b"")

        def sample(self, **kwargs):
            return 1

    engine = GGUFEngine("dummy.gguf")
    engine._llm = FakeLlama()
    engine._meta_cache = {
        "path": "dummy.gguf",
        "name": "fake-model",
        "architecture": "llama",
        "quant": "Q4_0",
        "n_vocab": 3,
        "n_ctx": 128,
        "size_bytes": 1024,
    }

    frames = list(engine.generate(prompt="hello", max_new_tokens=2))
    token_frames = [f for f in frames if f.get("type") == "token"]
    assert len(token_frames) == 2
    for tf in token_frames:
        assert "attention_focus" in tf
        assert tf["attention_focus"] == []
