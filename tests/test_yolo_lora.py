"""LoRA adapter unit tests (CPU, no ultralytics).

The real target is the YOLO-World SPPF stage; these tests exercise the same
code paths on a shape-faithful stand-in (Conv-BN-SiLU 1x1 in, maxpool
pyramid, 1x1 out — the ultralytics SPPF layout).
"""
import torch
from torch import nn

from microvla.perception.lora import (
    LoRAConv2d,
    base_parameters,
    clone_with_lora,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
)


class TinySPPF(nn.Module):
    def __init__(self, c1=8, c2=8):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(nn.Conv2d(c1, c_, 1, bias=False),
                                 nn.BatchNorm2d(c_), nn.SiLU())
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.cv2 = nn.Sequential(nn.Conv2d(c_ * 4, c2, 1, bias=False),
                                 nn.BatchNorm2d(c2), nn.SiLU())

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))


def _stage():
    torch.manual_seed(0)
    s = TinySPPF()
    # Give BN non-trivial running stats so eval-mode drift would be visible.
    s.train()
    for _ in range(3):
        s(torch.randn(4, 8, 6, 6))
    s.eval()
    return s


def test_identity_at_init():
    stage = _stage()
    adapted = clone_with_lora(stage, r=4)
    x = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        assert torch.allclose(stage(x), adapted(x), atol=1e-6)


def test_original_untouched_and_grads_reach_only_lora():
    stage = _stage()
    before = {k: v.clone() for k, v in stage.state_dict().items()}
    adapted = clone_with_lora(stage, r=4)
    out = adapted(torch.randn(2, 8, 6, 6)).sum()
    out.backward()
    for p in lora_parameters(adapted):
        assert p.requires_grad
    for name, p in adapted.named_parameters():
        if ".down." in name or ".up." in name:
            continue
        assert not p.requires_grad, name
    for k, v in stage.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_lora_changes_output_after_update():
    stage = _stage()
    adapted = clone_with_lora(stage, r=4, alpha=8.0)
    x = torch.randn(2, 8, 6, 6)
    opt = torch.optim.SGD(list(lora_parameters(adapted)), lr=1.0)
    (adapted(x).sum()).backward()
    opt.step()
    with torch.no_grad():
        assert not torch.allclose(stage(x), adapted(x), atol=1e-6)


def test_base_parameters_unfreeze_path():
    adapted = clone_with_lora(_stage(), r=2)
    base = list(base_parameters(adapted))
    assert base and all(not p.requires_grad for p in base)
    for p in base:
        p.requires_grad_(True)
    (adapted(torch.randn(1, 8, 6, 6)).sum()).backward()
    assert all(p.grad is not None for p in base)
    # BN running stats must not be trainable parameters.
    names = {n for n, p in adapted.named_parameters() if p.requires_grad}
    assert not any("running" in n for n in names)


def test_lora_state_dict_roundtrip():
    a = clone_with_lora(_stage(), r=4)
    for p in lora_parameters(a):
        with torch.no_grad():
            p.add_(torch.randn_like(p))
    sd = lora_state_dict(a)
    assert sd and all((".down." in k or ".up." in k) for k in sd)
    b = clone_with_lora(_stage(), r=4)
    load_lora_state_dict(b, sd)
    x = torch.randn(2, 8, 6, 6)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


def test_wrapped_conv_count():
    adapted = clone_with_lora(_stage(), r=2)
    n = sum(1 for m in adapted.modules() if isinstance(m, LoRAConv2d))
    assert n == 2  # cv1 + cv2
