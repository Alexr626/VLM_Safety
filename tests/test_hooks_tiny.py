"""model_internals + hooks on tiny randomly initialised HF models (CPU, no downloads).

Checks each hook site against `output_hidden_states` so the layer-index
convention in src/model_internals.py stays true across transformers versions.
"""

from types import SimpleNamespace

import pytest
import torch

from src.hooks import capture_residual, edit_residual
from src.model import LLaVAWrapper, Qwen2VLWrapper
from src.model_internals import get_internals

D, N_LAYERS, VOCAB = 64, 3, 128


def _tiny_llava():
    from transformers import (CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration)
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                                       num_hidden_layers=1, num_attention_heads=2,
                                       image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=D, intermediate_size=128,
                                num_hidden_layers=N_LAYERS, num_attention_heads=4,
                                num_key_value_heads=4, vocab_size=VOCAB),
        image_token_index=VOCAB - 1,
    )
    w = LLaVAWrapper()
    w.model = LlavaForConditionalGeneration(cfg).eval()
    w.tokenizer = SimpleNamespace(bos_token_id=1)
    return w


def _tiny_qwen():
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    cfg = Qwen2_5_VLConfig(
        hidden_size=D, intermediate_size=128, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 2, "out_hidden_size": D,
                       "fullatt_block_indexes": [0]},
        image_token_id=VOCAB - 1, video_token_id=VOCAB - 2,
        vision_start_token_id=VOCAB - 3, vision_end_token_id=VOCAB - 4,
    )
    w = Qwen2VLWrapper(torch_dtype=torch.float32)
    w.model = Qwen2_5_VLForConditionalGeneration(cfg).eval()
    w.tokenizer = SimpleNamespace(bos_token_id=None)
    return w


@pytest.fixture(params=["llava", "qwen2_vl"])
def wrapper(request):
    torch.manual_seed(0)
    return _tiny_llava() if request.param == "llava" else _tiny_qwen()


def _ids():
    return torch.tensor([[1, 5, 9, 17, 33, 42]])


def _hidden_states(w, ids):
    with torch.no_grad():
        return w.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states


def test_internals_resolve(wrapper):
    mi = get_internals(wrapper)
    assert mi.num_layers == N_LAYERS
    gain, eps = mi.rmsnorm_params(0)
    assert gain.shape == (D,) and eps > 0
    q, k = mi.qk_proj(1)
    assert q.in_features == D and k.in_features == D
    assert (q.bias is not None) == (mi.family == "qwen2_vl")
    assert mi.lm_head.out_features == VOCAB
    assert mi.bos_position(_ids()) == (0 if mi.family == "llava" else None)


@pytest.mark.parametrize("layer", [0, N_LAYERS - 1])
def test_capture_sites_match_hidden_states(wrapper, layer):
    mi = get_internals(wrapper)
    ids = _ids()
    hs = _hidden_states(wrapper, ids)
    for site, expected in (("input", hs[layer]), ("output", hs[layer + 1])):
        with capture_residual(mi, [layer], site=site, positions="all") as store:
            _hidden_states(wrapper, ids)
        got = store[layer][0]
        if site == "output" and layer == N_LAYERS - 1:
            # HF applies the final norm to the last entry of hidden_states.
            got = mi.final_norm(got)
        torch.testing.assert_close(got, expected.float(), atol=1e-4, rtol=1e-4)
    with capture_residual(mi, [layer], site="normed", positions="last") as store:
        _hidden_states(wrapper, ids)
    with torch.no_grad():
        expected = mi.input_layernorm(layer)(hs[layer])[:, -1, :]
    torch.testing.assert_close(store[layer][0], expected.float(), atol=1e-4, rtol=1e-4)


def test_edit_residual_shifts_next_layer_input(wrapper):
    mi = get_internals(wrapper)
    ids = _ids()
    base = _hidden_states(wrapper, ids)
    delta = torch.randn(D)

    def edit(h, call_idx):
        h = h.clone()
        h[:, -1, :] += delta.to(h.dtype)
        return h

    with edit_residual(mi, 0, edit, site="output"):
        edited = _hidden_states(wrapper, ids)
    torch.testing.assert_close(edited[1][0, -1] - base[1][0, -1], delta, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(edited[1][0, :-1], base[1][0, :-1])

    with edit_residual(mi, 1, edit, site="input"):
        edited_in = _hidden_states(wrapper, ids)
    torch.testing.assert_close(edited_in[2], edited[2], atol=1e-4, rtol=1e-4)


def test_capture_during_generate_counts_calls(wrapper):
    mi = get_internals(wrapper)
    ids = _ids()
    with capture_residual(mi, [1], site="output") as store:
        with torch.no_grad():
            wrapper.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   max_new_tokens=4, min_new_tokens=4, do_sample=False)
    assert len(store[1]) == 4  # prefill + 3 decode steps
    assert all(t.shape == (1, D) for t in store[1])
