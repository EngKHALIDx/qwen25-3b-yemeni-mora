import sys
from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "peft-mora" / "src"))

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


class TinyConfig(PretrainedConfig):
    model_type = "tiny-gl-log-mora"

    def __init__(self, hidden_size=8, vocab_size=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class TinyCausalLM(PreTrainedModel):
    config_class = TinyConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.q_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size).float()
        hidden = torch.tanh(self.q_proj(hidden))
        hidden = torch.tanh(self.v_proj(hidden))
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return CausalLMOutput(loss=loss, logits=logits)


def main():
    torch.manual_seed(13)
    model = TinyCausalLM(TinyConfig())
    config = LoraConfig(
        r=2,
        lora_alpha=2,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        use_mora=True,
        mora_type=1,
        use_gl_log_mora=True,
        gl_log_lambda=0.01,
        gl_log_delta=1e-3,
    )
    model = get_peft_model(model, config)
    gl_layers = [m for m in model.modules() if getattr(m, "use_gl_log_mora", {}).get("default", False)]
    assert len(gl_layers) == 2, len(gl_layers)
    assert all(m.lora_Q["default"].requires_grad for m in gl_layers)

    input_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    labels = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 3]])
    output = model(input_ids=input_ids, labels=labels)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert all(m.lora_Q["default"].grad is not None for m in gl_layers)

    state = get_peft_model_state_dict(model)
    q_keys = [key for key in state if "lora_Q" in key]
    assert len(q_keys) == 2, q_keys

    reloaded = get_peft_model(
        TinyCausalLM(TinyConfig()),
        config,
    )
    set_peft_model_state_dict(reloaded, state)
    for original, restored in zip(
        [m.lora_Q["default"] for m in gl_layers],
        [m.lora_Q["default"] for m in reloaded.modules() if getattr(m, "use_gl_log_mora", {}).get("default", False)],
    ):
        assert torch.allclose(original, restored)
    print(f"integration passed: {len(gl_layers)} GL layers; {len(q_keys)} Q tensors saved and reloaded")


if __name__ == "__main__":
    main()
