"""The three baseline architectures (FINAL_research_problem.md, section 7).

All take a raw waveform batch (B, 16000) and return logits (B, 3). The log-mel model
contains its own float front-end; only the backbone after it will be quantized.

Parameter counts (verify with `python -m pcg.models`):
  tiny_cnn   ~ 27 k   raw waveform, De Simone-style light CNN (edge target)
  m5         ~ 357 k  raw waveform, Dai et al. 2017 M5 (tests H1: no log compression)
  logmel_cnn ~ 308 k  log-mel + small 2D ResNet-style CNN (Challenge-winner style)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import LogMel, SpecAugment

N_CLASSES = 3


def conv_bn_relu_1d(cin: int, cout: int, k: int, s: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv1d(cin, cout, k, s, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True))


class TinyCNN(nn.Module):
    """~25k params. Strided convs instead of pooling where possible (MCU friendly)."""

    def __init__(self, n_classes: int = N_CLASSES, width: int = 12):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            conv_bn_relu_1d(1, w, 31, 4),      # 16000 -> 4000
            nn.MaxPool1d(4),                    # -> 1000
            conv_bn_relu_1d(w, 2 * w, 15, 2),   # -> 500
            nn.MaxPool1d(2),                    # -> 250
            conv_bn_relu_1d(2 * w, 4 * w, 9, 2),   # -> 125
            conv_bn_relu_1d(4 * w, 4 * w, 5, 1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * w, n_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav.unsqueeze(1))


class M5(nn.Module):
    """Dai et al. 2017 'Very deep CNNs for raw waveforms', M5 variant (n_channel 128)."""

    def __init__(self, n_classes: int = N_CLASSES, n_channel: int = 128, stride: int = 16):
        super().__init__()
        c = n_channel
        self.net = nn.Sequential(
            nn.Conv1d(1, c, 80, stride=stride, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            conv_bn_relu_1d(c, c, 3), nn.MaxPool1d(4),
            conv_bn_relu_1d(c, 2 * c, 3), nn.MaxPool1d(4),
            conv_bn_relu_1d(2 * c, 2 * c, 3), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(2 * c, n_classes),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav.unsqueeze(1))


class BasicBlock2d(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        # one ReLU module per use: the PT2E x86 quantizer partitions by module instance and
        # rejects a module that produces two outputs
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idn = x if self.short is None else self.short(x)
        out = self.relu1(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return self.relu2(out + idn)


class LogMelCNN(nn.Module):
    """Float log-mel front-end + small ResNet-style 2D backbone (~0.4M params)."""

    def __init__(self, n_classes: int = N_CLASSES, widths=(16, 32, 64, 128), specaugment: bool = True):
        super().__init__()
        self.frontend = LogMel()
        self.specaug = SpecAugment() if specaugment else nn.Identity()
        w0, w1, w2, w3 = widths
        self.backbone = nn.Sequential(
            nn.Conv2d(1, w0, 3, 1, 1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True),
            BasicBlock2d(w0, w0), BasicBlock2d(w0, w1, 2), BasicBlock2d(w1, w2, 2), BasicBlock2d(w2, w3, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w3, n_classes),
        )

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        return self.specaug(self.frontend(wav))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.features(wav))


class EffNetB0LogMel(LogMelCNN):
    """Log-mel front-end + torchvision EfficientNet-B0 (ImageNet-pretrained, 1-channel stem, 3-way head).

    Included because it appears in both 2026 PCG papers and is the architecture reported as most
    quantization-fragile (Rabbi 2026, images) and as collapsing under INT8 PTQ on CirCor
    (Ludera & Kersten-Oertel, MICCAI 2026). ~4.0M params.
    """

    def __init__(self, n_classes: int = N_CLASSES, pretrained: bool = True, specaugment: bool = True):
        nn.Module.__init__(self)
        from torchvision.models import efficientnet_b0
        self.frontend = LogMel()
        self.specaug = SpecAugment() if specaugment else nn.Identity()
        net = efficientnet_b0(weights="DEFAULT" if pretrained else None)
        stem = net.features[0][0]  # Conv2d(3, 32, 3, stride 2)
        new_stem = nn.Conv2d(1, stem.out_channels, stem.kernel_size, stem.stride, stem.padding, bias=False)
        with torch.no_grad():
            new_stem.weight.copy_(stem.weight.sum(dim=1, keepdim=True))
        net.features[0][0] = new_stem
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)
        self.backbone = net


MODELS = {"tiny_cnn": TinyCNN, "m5": M5, "logmel_cnn": LogMelCNN, "effnet_b0": EffNetB0LogMel}


def build_model(name: str) -> nn.Module:
    return MODELS[name]()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    x = torch.randn(2, 16000)
    for name in MODELS:
        m = build_model(name).eval()
        with torch.no_grad():
            y = m(x)
        print(f"{name:12s} params={n_params(m):>9,d} out={tuple(y.shape)}")
