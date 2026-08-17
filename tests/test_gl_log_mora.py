import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "peft-mora" / "src"))

from peft.tuners.lora.layer import Linear


def build_layer(in_features=7, out_features=9, mora_type=1):
    torch.manual_seed(7)
    base = torch.nn.Linear(in_features, out_features, bias=True)
    layer = Linear(
        base,
        "default",
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        init_lora_weights=True,
        use_mora=True,
        mora_type=mora_type,
        use_gl_log_mora=True,
        gl_log_lambda=0.01,
        gl_log_delta=1e-3,
    )
    with torch.no_grad():
        layer.lora_A["default"].weight.normal_(mean=0.0, std=0.2)
        layer.lora_Q["default"].copy_(torch.eye(layer.r["default"]))
    return layer


def test_identity_initialization_and_regularizer():
    layer = build_layer()
    q = layer.lora_Q["default"]
    assert q.shape == (layer.r["default"], layer.r["default"])
    assert torch.allclose(q, torch.eye(layer.r["default"]), atol=1e-6)
    expected = -0.01 * layer.r["default"] * math.log(1.001)
    actual = layer.gl_log_mora_regularizer("default").item()
    assert abs(actual - expected) < 1e-6


def test_forward_matches_reconstructed_delta():
    layer = build_layer()
    x = torch.randn(5, layer.in_features)
    with torch.no_grad():
        delta = layer.get_delta_weight("default")
        explicit = layer._apply_mora(
            x,
            layer.lora_A["default"],
            layer.lora_B["default"],
            layer.scaling["default"],
            "default",
        )
        reconstructed = torch.nn.functional.linear(x, delta)
    assert torch.allclose(explicit, reconstructed, atol=2e-5, rtol=2e-5)


def test_merge_and_unmerge_round_trip():
    layer = build_layer()
    x = torch.randn(3, layer.in_features)
    with torch.no_grad():
        before_weight = layer.base_layer.weight.detach().clone()
        before_bias = layer.base_layer.bias.detach().clone()
        unmerged_output = layer(x)
        layer.merge(safe_merge=True)
        merged_output = layer(x)
        layer.unmerge()
        restored_output = layer(x)
    assert torch.allclose(unmerged_output, merged_output, atol=3e-5, rtol=3e-5)
    assert torch.allclose(unmerged_output, restored_output, atol=3e-5, rtol=3e-5)
    assert torch.allclose(layer.base_layer.weight, before_weight, atol=3e-5, rtol=3e-5)
    assert torch.allclose(layer.base_layer.bias, before_bias, atol=3e-5, rtol=3e-5)


def test_q_and_m_receive_gradients():
    layer = build_layer()
    x = torch.randn(4, layer.in_features)
    # M starts at zero in the actual initialization; its task gradient must still flow.
    with torch.no_grad():
        layer.lora_A["default"].weight.zero_()
    output = layer(x)
    loss = output.square().mean() + layer.gl_log_mora_regularizer("default")
    loss.backward()
    assert layer.lora_Q["default"].grad is not None
    assert layer.lora_A["default"].weight.grad is not None
    assert torch.isfinite(layer.lora_Q["default"].grad).all()
    assert torch.isfinite(layer.lora_A["default"].weight.grad).all()


if __name__ == "__main__":
    test_identity_initialization_and_regularizer()
    test_forward_matches_reconstructed_delta()
    test_merge_and_unmerge_round_trip()
    test_q_and_m_receive_gradients()
    print("GL-log-MoRA unit tests passed")
