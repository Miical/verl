"""Utility helpers to register custom VLA models with Hugging Face Auto classes."""

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .openvla_oft.configuration_prismatic import OpenVLAConfig
from .openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
from .openvla_oft.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from .pi0_torch import PI0TorchConfig, PI0TorchImageProcessor, PI0TorchModel, PI0TorchProcessor

_REGISTERED_MODELS = {
    "openvla_oft": False,
    "pi0_torch": False,
}

def register_openvla_oft() -> None:
    """Register the OpenVLA OFT model and processors."""
    if _REGISTERED_MODELS["openvla_oft"]:
        return

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    _REGISTERED_MODELS["openvla_oft"] = True


def register_pi0_torch_model() -> None:
    """Register the PI0 wrapper with the HF auto classes."""
    if _REGISTERED_MODELS["pi0_torch"]:
        return

    AutoConfig.register("pi0_torch", PI0TorchConfig)
    AutoImageProcessor.register(PI0TorchConfig, PI0TorchImageProcessor)
    AutoProcessor.register(PI0TorchConfig, PI0TorchProcessor)
    AutoModelForVision2Seq.register(PI0TorchConfig, PI0TorchModel)

    _REGISTERED_MODELS["pi0_torch"] = True


def register_vla_models() -> None:
    """Register all custom VLA models with Hugging Face."""
    register_openvla_oft()
    register_pi0_torch_model()
