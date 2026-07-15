import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTForImageClassification
import timm

def build_vit_base_custom_head(num_classes, dropout=0.30, hidden_ratio=0.5):
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    model.config.num_labels = num_classes
    model.num_labels = num_classes

    hidden = model.config.hidden_size
    mid = int(hidden * hidden_ratio)

    model.classifier = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, num_classes),
    )
    return model

class ConvNeXtCustom(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout=0.30, hidden_ratio=0.5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        self.hidden = getattr(self.backbone, "num_features", None)
        if self.hidden is None:
            raise ValueError("Could not find backbone.num_features (unexpected timm model).")

        mid = int(self.hidden * hidden_ratio)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_classes)
        )

    def forward(self, x):
        feats = self.backbone.forward_features(x)

        if feats.dim() == 4:
            if feats.shape[1] != self.hidden and feats.shape[-1] == self.hidden:
                feats = feats.permute(0, 3, 1, 2).contiguous()
            feats = self.pool(feats).flatten(1)
        elif feats.dim() == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

        return self.head(feats)

def build_convnext_custom_head(num_classes, variant="tiny", dropout=0.30, hidden_ratio=0.5, pretrained=True):
    model_name = "convnext_tiny" if variant == "tiny" else "convnext_base"
    return ConvNeXtCustom(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
        pretrained=pretrained
    )        