"""Model factory for change-detection experiments."""
from .fc import FCSiam
from .siamese import SiameseUNet


def build_model(name, pretrained=True):
    name = name.lower()
    # classic from-scratch baselines
    if name == "fc_ef":
        return FCSiam(mode="ef")
    if name == "fc_siam_diff":
        return FCSiam(mode="diff")
    if name == "fc_siam_conc":
        return FCSiam(mode="conc")
    # backbone baselines
    if name == "resnet18_siam":
        return SiameseUNet(backbone="resnet18", fusion="diff", pretrained=pretrained)
    # ---- ours ----
    if name == "lightcd":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="tda",
                           use_ca=True, use_sa=True, pretrained=pretrained)
    # higher-resolution variant: less aggressive downsampling (output_stride=16)
    if name == "lightcd_hr":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="tda",
                           use_ca=True, use_sa=True, pretrained=pretrained, output_stride=16)
    # higher-capacity variant: MobileNetV3-Large backbone (still edge-efficient)
    if name == "lightcd_large":
        return SiameseUNet(backbone="mobilenetv3_large_100", fusion="tda",
                           use_ca=True, use_sa=True, pretrained=pretrained)
    # ---- ablations of ours ----
    if name == "lightcd_concat":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="concat", pretrained=pretrained)
    if name == "lightcd_diff":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="diff", pretrained=pretrained)
    if name == "lightcd_noca":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="tda",
                           use_ca=False, use_sa=True, pretrained=pretrained)
    if name == "lightcd_nosa":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="tda",
                           use_ca=True, use_sa=False, pretrained=pretrained)
    # [f_A ; f_B ; d] fused, no attention: separates the effect of d from the gates
    if name == "lightcd_nogate":
        return SiameseUNet(backbone="mobilenetv3_small_100", fusion="tda",
                           use_ca=False, use_sa=False, pretrained=pretrained)
    # published lightweight / transformer CD baselines, trained under our protocol
    if name == "tinycd":
        from .tinycd import TinyCD
        return TinyCD(pretrained=pretrained)
    if name == "a2net":
        from .a2net import A2Net
        return A2Net(pretrained=pretrained)
    if name == "bit":
        from .bit import BIT
        return BIT(pretrained=pretrained)
    raise ValueError(f"unknown model {name}")


MODEL_NAMES = [
    "fc_ef", "fc_siam_diff", "fc_siam_conc", "resnet18_siam",
    "lightcd", "lightcd_hr", "lightcd_large",
    "lightcd_concat", "lightcd_diff", "lightcd_noca", "lightcd_nosa", "lightcd_nogate",
    "tinycd", "a2net", "bit",
]
