from __future__ import annotations

from torch import nn
from torchvision import models


MODEL_DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "resnext50_32x4d": "ResNeXt-50 32x4d",
    "seresnet50": "SE-ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
}


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "convnext_tiny":
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        )
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name == "seresnet50":
        import timm

        return timm.create_model(
            "seresnet50",
            pretrained=pretrained,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
