"""
ReFusion: multimodal reliability fusion network.
Dual-domain encoders feed per-modality classifiers and an uncertainty-aware gate,
with hybrid fusion and an optional Refiner.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .layers import ModalityClassifier

_EPS = 1e-8


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


def grl(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor, lambda_: Optional[float] = None) -> torch.Tensor:
        return grl(x, lambda_ if lambda_ is not None else self.lambda_)


# ---------- Domain discriminator ----------
class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ---------- Time-domain multi-scale branch ----------
class TimeBranchMultiScale(nn.Module):
    def __init__(self, out_channels_per_scale: int = 24, dilation: int = 2):
        super().__init__()
        kernels = (3, 7, 15)
        self.branches = nn.ModuleList()
        for k in kernels:
            padding = ((k - 1) * dilation) // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(1, out_channels_per_scale, k, padding=padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels_per_scale),
                    nn.ReLU(),
                )
            )
        self.out_channels = out_channels_per_scale * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return torch.cat([branch(x) for branch in self.branches], dim=1)


# ---------- Frequency-domain branch ----------
class FreqBranch(nn.Module):
    def __init__(self, fft_size: int, out_channels: int = 32, hidden: int = 64):
        super().__init__()
        self.fft_size = fft_size
        freq_len = fft_size // 2 + 1
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, 7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(freq_len // 4),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden * (freq_len // 4), out_channels),
            nn.ReLU(),
        )
        self._out_dim = out_channels

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(x, dim=-1).abs()
        spec = spec.unsqueeze(1)
        out = self.conv(spec)
        out = out.flatten(1)
        return self.fc(out)


# ---------- Squeeze-excitation (inside encoder) ----------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


# ---------- Uncertainty-aware reliability gate ----------
class UncertaintyAwareReliabilityGate(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_temperature: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = latent_dim + num_classes + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.use_temperature = use_temperature
        if use_temperature:
            self.log_T = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1).clamp(min=_EPS)
        return -(p * p.log()).sum(dim=-1)


# ---------- Prototype loss ----------
def prototype_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    prototypes: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, D = embedding.shape
    if prototypes is not None:
        proto_y = prototypes[labels.clamp(0, num_classes - 1)]
        return F.mse_loss(embedding, proto_y, reduction="mean"), None
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes=num_classes).float()
    count = one_hot.sum(dim=0, keepdim=True).clamp(min=1e-8)
    prototypes_batch = torch.mm(one_hot.t(), embedding) / count.t()
    proto_y = prototypes_batch[labels.clamp(0, num_classes - 1)]
    loss = F.mse_loss(embedding, proto_y, reduction="mean")
    return loss, prototypes_batch.detach()


# ---------- Main model ----------
class ReFusion(nn.Module):
    """
    Multimodal fault diagnosis: per-modality dual-domain encoding, per-modality heads,
    reliability gating, and fused prediction.
    """

    def __init__(
        self,
        modalities: List[str],
        input_size: int = 1024,
        latent_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.15,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        use_hybrid_fusion: bool = True,
        use_refiner: bool = True,
        prob_weight: float = 0.6,
        feat_weight: float = 0.4,
        temperature: float = 0.7,
        learnable_temperature: bool = True,
        use_bad_modality: bool = False,
        use_domain_adaptation: bool = False,
    ):
        super().__init__()
        self.modalities = modalities
        self.num_classes = num_classes
        self.use_hybrid_fusion = use_hybrid_fusion
        self.prob_weight = prob_weight
        self.feat_weight = feat_weight
        self.use_bad_modality = use_bad_modality
        self.use_domain_adaptation = use_domain_adaptation

        self.encoders = nn.ModuleDict({
            m: DualDomainEncoder(
                input_size=input_size,
                latent_dim=latent_dim,
                dropout=dropout,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
            )
            for m in modalities
        })
        self.classifiers = nn.ModuleDict({
            m: ModalityClassifier(latent_dim, num_classes) for m in modalities
        })
        self.reliability_gate = UncertaintyAwareReliabilityGate(
            latent_dim=latent_dim,
            num_classes=num_classes,
            hidden_dim=64,
            dropout=dropout * 0.5,
            use_temperature=True,
        )
        if use_hybrid_fusion:
            self.feature_fusion_classifier = ModalityClassifier(latent_dim, num_classes)
        else:
            self.feature_fusion_classifier = None
        if use_refiner:
            self.refiner = nn.Sequential(
                nn.Linear(num_classes, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, num_classes),
            )
        else:
            self.refiner = None
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(max(temperature, 1e-8))))
        else:
            self.register_buffer("_temperature_val", torch.tensor(temperature))
            self.log_temperature = None
        if use_domain_adaptation:
            self.domain_discriminators = nn.ModuleDict({
                m: DomainDiscriminator(latent_dim, hidden=64, dropout=dropout * 0.5)
                for m in modalities
            })
        else:
            self.domain_discriminators = None

    @property
    def temperature(self) -> torch.Tensor:
        if self.log_temperature is not None:
            return torch.exp(self.log_temperature).clamp(min=1e-8)
        return self._temperature_val.clamp(min=1e-8)

    def forward(
        self,
        modalities_dict: Dict[str, torch.Tensor],
        prev_modalities_dict: Dict = None,
        compute_reliability: bool = True,
        corrupted_modality_indices: Optional[torch.Tensor] = None,
        corruption_gamma: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        domain_grl_lambda: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        logits_all = []
        h_prime_list = []
        for m in self.modalities:
            z = self.encoders[m](modalities_dict[m])
            h_prime = z
            logits = self.classifiers[m](h_prime)
            h_prime_list.append(h_prime)
            logits_all.append(logits)

        logits_stack = torch.stack(logits_all, dim=0)
        gates_stack = self.reliability_gate(h_prime_list, logits_all)
        gates_norm = gates_stack

        if self.use_hybrid_fusion:
            h_fused = sum(
                gates_norm[i].mean(dim=1, keepdim=True) * h_prime_list[i]
                for i in range(len(h_prime_list))
            )
            feature_logits = self.feature_fusion_classifier(h_fused)
        else:
            h_fused = sum(h_prime_list) / len(h_prime_list)
            feature_logits = None

        temp = self.temperature
        if temp.dim() == 0:
            temp = temp.item()
        logits_scaled = logits_stack / max(float(temp), 1e-8)
        final_logits = (gates_norm * logits_scaled).sum(dim=0)
        final_probs = F.softmax(final_logits, dim=1)

        if self.use_hybrid_fusion and feature_logits is not None:
            final_logits = self.prob_weight * final_logits + self.feat_weight * feature_logits
            final_probs = F.softmax(final_logits, dim=1)
        if self.refiner is not None:
            refined = self.refiner(final_logits)
            final_logits = final_logits + refined
            final_probs = F.softmax(final_logits, dim=1)

        single_modality_logits = {m: logits_all[i] for i, m in enumerate(self.modalities)}
        single_modality_predictions = {
            m: F.softmax(logits_all[i], dim=1) for i, m in enumerate(self.modalities)
        }
        reliability_scores = {
            m: gates_norm[i].mean(dim=1) for i, m in enumerate(self.modalities)
        }
        reliability_raw = {m: gates_stack[i].mean(dim=1) for i, m in enumerate(self.modalities)}
        out = {
            "final_logits": final_logits,
            "final_fusion": final_probs,
            "single_modality_logits": single_modality_logits,
            "single_modality_predictions": single_modality_predictions,
            "reliability_scores": reliability_scores,
            "reliability_raw": reliability_raw,
            "gates": gates_norm.mean(dim=2),
            "gates_per_class": gates_norm,
            "embedding": h_fused,
        }
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    modalities = ["a", "b", "c"]
    model = ReFusion(
        modalities=modalities,
        input_size=1024,
        latent_dim=128,
        num_classes=5,
        use_hybrid_fusion=True,
        use_refiner=True,
        use_bad_modality=True,
        use_domain_adaptation=True,
    )
    x = {m: torch.randn(4, 1024) for m in modalities}
    out = model(x)
    print("ReFusion final_logits:", out["final_logits"].shape)
    print("embedding:", out["embedding"].shape)
    print("Params:", f"{count_parameters(model):,}")

    emb = out["embedding"]
    lab = torch.tensor([0, 1, 2, 0])
    l_proto, _ = prototype_loss(emb, lab, num_classes=5)
    print("L_proto:", l_proto.item())
