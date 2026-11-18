"""torch.compile on a booted bridge: one graph, same outputs, hooks still fire."""

import pytest
import torch
import torch._dynamo
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.model_bridge import TransformerBridge

TINY = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    vocab_size=100,
    max_position_embeddings=64,
)
# (model_type, extra config, name boot uses for its tokenizer)
ARCHS = {
    "qwen2": ({"num_key_value_heads": 2}, "Qwen/Qwen2-0.5B"),
    "llama": ({"num_key_value_heads": 2}, "Qwen/Qwen2-0.5B"),
    "gpt2": ({"n_embd": 32, "n_layer": 2, "n_head": 4, "n_positions": 64}, "gpt2"),
    "mixtral": ({"num_key_value_heads": 2, "num_local_experts": 4}, "Qwen/Qwen2-0.5B"),
}


def _bridge(model_type: str) -> TransformerBridge:
    extra, name = ARCHS[model_type]
    cfg = AutoConfig.for_model(model_type, **{**TINY, **extra})
    torch.manual_seed(0)
    hf_model = AutoModelForCausalLM.from_config(cfg).eval()
    return TransformerBridge.boot_transformers(name, hf_model=hf_model, device="cpu")


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


TOKENS = torch.tensor([[1, 5, 9, 2, 7, 3]])


@pytest.mark.parametrize("model_type", list(ARCHS))
def test_first_forward_compiles_without_graph_breaks(model_type):
    bridge = _bridge(model_type)
    with torch.no_grad():
        explanation = torch._dynamo.explain(bridge.original_model)(TOKENS)
    assert explanation.graph_break_count == 0, explanation.break_reasons


def test_compiled_logits_hooks_and_cache_match_eager():
    bridge = _bridge("qwen2")
    with torch.no_grad():
        eager_logits = bridge(TOKENS)
        _, eager_cache = bridge.run_with_cache(TOKENS)
        # The HF model is what generate() compiles; the bridge's own forward wraps it.
        bridge.original_model.compile(backend="aot_eager")
        torch.testing.assert_close(bridge(TOKENS), eager_logits)

        bridge.blocks[0].hook_out.add_hook(lambda t, hook: torch.zeros_like(t))
        hooked = bridge(TOKENS)
        assert not torch.allclose(hooked, eager_logits)
        bridge.reset_hooks()
        torch.testing.assert_close(bridge(TOKENS), eager_logits)

        _, cache = bridge.run_with_cache(TOKENS)
    assert set(cache.keys()) == set(eager_cache.keys())
    for name in eager_cache.keys():
        torch.testing.assert_close(cache[name], eager_cache[name])
