"""boot()'s opt-in ``attn_implementation="sdpa"``: SDPA on unhooked calls, eager when observed."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache
from transformers.models.qwen2 import modeling_qwen2

from transformer_lens.model_bridge import TransformerBridge

# One PositionEmbeddingsAttentionBridge architecture (GQA) and one base AttentionBridge one.
MODELS = {
    "qwen2": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        dict(num_attention_heads=4, num_key_value_heads=2, intermediate_size=64),
    ),
    "opt": ("facebook/opt-125m", dict(ffn_dim=64, word_embed_proj_dim=32)),
}
TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, vocab_size=100)
HOOKS = ("blocks.0.attn.hook_pattern", "blocks.1.attn.hook_attn_scores")


def _tiny(kind, attn_implementation):
    torch.manual_seed(0)
    config = AutoConfig.for_model(kind, **{**TINY, **MODELS[kind][1]})
    return AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation).eval()


def _bridge(kind, **kwargs):
    return TransformerBridge.boot_transformers(
        MODELS[kind][0], hf_model=_tiny(kind, "sdpa"), device="cpu", **kwargs
    )


def _inputs():
    torch.manual_seed(1)
    tokens = torch.randint(0, 100, (2, 7))
    padded = torch.ones(2, 7, dtype=torch.long)
    padded[1, :3] = 0
    return tokens, padded


def _hf_logits(model, tokens, mask):
    # The bridge derives positions from the padding mask.
    positions = None if mask is None else (mask.cumsum(-1) - 1).clamp(min=0)
    return model(tokens, attention_mask=mask, position_ids=positions).logits


def _count_sdpa_calls(monkeypatch):
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def counting(*args, **kwargs):
        calls.append(1)
        return sdpa(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", counting)
    return calls


@pytest.mark.parametrize("kind", MODELS)
def test_default_boot_stays_eager(kind, monkeypatch):
    bridge = _bridge(kind)
    hf_eager = _tiny(kind, "eager")
    tokens, padded = _inputs()
    calls = _count_sdpa_calls(monkeypatch)

    with torch.no_grad():
        for mask in (None, padded):
            out = bridge(tokens, attention_mask=mask, return_type="logits")
            ref = _hf_logits(hf_eager, tokens, mask)
            keep = slice(None) if mask is None else mask.bool()
            assert torch.equal(out[keep], ref[keep])

    assert bridge.original_model.config._attn_implementation == "eager"
    assert not calls


@pytest.mark.parametrize("kind", MODELS)
def test_sdpa_without_attention_hooks_matches_hf_sdpa(kind, monkeypatch):
    bridge = _bridge(kind, attn_implementation="sdpa")
    hf_sdpa = _tiny(kind, "sdpa")
    tokens, padded = _inputs()

    with torch.no_grad():
        for mask in (None, padded):
            ref = _hf_logits(hf_sdpa, tokens, mask)
            calls = _count_sdpa_calls(monkeypatch)
            out = bridge(tokens, attention_mask=mask, return_type="logits")
            assert len(calls) == 2
            keep = slice(None) if mask is None else mask.bool()
            torch.testing.assert_close(out[keep], ref[keep], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("kind", MODELS)
def test_sdpa_falls_back_to_eager_when_scores_or_pattern_are_hooked(kind):
    bridge = _bridge(kind, attn_implementation="sdpa")
    eager_bridge = _bridge(kind)
    tokens, padded = _inputs()

    for mask in (None, padded):
        _, cache = bridge.run_with_cache(tokens, attention_mask=mask)
        _, ref = eager_bridge.run_with_cache(tokens, attention_mask=mask)
        for name in HOOKS:
            assert torch.equal(cache[name], ref[name])

    seen = []
    bridge.blocks[0].attn.hook_pattern.add_hook(lambda t, hook: seen.append(t.shape))
    bridge(tokens)
    assert seen == [(2, 4, 7, 7)]
    assert bridge.original_model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("kind", MODELS)
def test_sdpa_cached_generation_matches_hf_sdpa(kind):
    bridge = _bridge(kind, attn_implementation="sdpa")
    tokens, _ = _inputs()

    expected = _tiny(kind, "sdpa").generate(tokens[:1], max_new_tokens=8, do_sample=False)
    out = bridge.generate(
        tokens[:1],
        max_new_tokens=8,
        do_sample=False,
        use_past_kv_cache=True,
        return_type="tokens",
        verbose=False,
    )

    assert torch.equal(out, expected)


@pytest.mark.parametrize("hook_pattern", [False, True], ids=["sdpa", "eager-fallback"])
def test_sdpa_none_mask_over_padded_static_keys_matches_hf_sdpa(hook_pattern, monkeypatch):
    # For a static cache's prompt HF can pass no mask and rely on SDPA's top-left causal alignment
    # over the padded (still empty) keys. Reproduce that mask for both models.
    monkeypatch.setattr(modeling_qwen2, "create_causal_mask", lambda *args, **kwargs: None)
    bridge = _bridge("qwen2", attn_implementation="sdpa")
    if hook_pattern:
        bridge.add_hook("blocks.0.attn.hook_pattern", lambda pattern, hook: None)
    tokens = _inputs()[0][:1]

    def logits(model):
        cache = StaticCache(config=model.config, max_cache_len=tokens.shape[1] + 4)
        return model(tokens, past_key_values=cache, use_cache=True).logits

    torch.testing.assert_close(logits(bridge.original_model), logits(_tiny("qwen2", "sdpa")))


def test_unknown_attn_implementation_raises():
    with pytest.raises(ValueError, match="attn_implementation"):
        _bridge("opt", attn_implementation="flash_attention_2")
