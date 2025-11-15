# utils/models.py
import torch
import torch.nn as nn
from torchvision import models

def _add_head_dropout(model, p=0.2):
    """
    Inserts a Dropout layer before the classifier head.
    Works for common torchvision models without breaking pretrained weights.
    """

    # ResNet family
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Sequential(
            nn.Dropout(p),
            model.fc
        )
        return model

    # VGG family
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            new_layers = list(clf)
            new_layers.insert(-1, nn.Dropout(p))  # before last Linear
            model.classifier = nn.Sequential(*new_layers)
            return model

    # EfficientNetV2
    if hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            new_layers = list(clf)
            new_layers.insert(0, nn.Dropout(p))
            model.classifier = nn.Sequential(*new_layers)
            return model

    # ConvNeXt
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Sequential(
            nn.Dropout(p),
            model.classifier
        )
        return model

    print("[WARN] Dropout not injected for this model architecture.")
    return model


def get_pretrained_model(model_name: str,
                         dropout_p: float = 0.2,
                         device: torch.device = torch.device("cpu")):
    """
    Loads a pretrained model by name, injects dropout, and returns it with required input size.
    Returns:
        model, input_image_size
    """
    name = model_name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        img_size = 224

    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        img_size = 224

    elif name in ["vgg16_bn", "vgg16"]:
        weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if name == "vgg16_bn" else models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16_bn(weights=weights) if name == "vgg16_bn" else models.vgg16(weights=weights)
        img_size = 224

    elif name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        img_size = 384  # default for effnet-v2-s

    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        img_size = 224

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Inject dropout for MC Dropout
    if dropout_p > 0:
        model = _add_head_dropout(model, p=dropout_p)

    model = model.to(device)
    model.eval()
    return model, img_size
