"""Model registry for the Deep Generative Models package."""

from .base import GenerativeModel
from .gan import GenerativeAdversarialNetwork
from .vae import VariationalAutoEncoder
from .flow import NormalizingFlowModel

MODEL_REGISTRY = {
    "gan": GenerativeAdversarialNetwork,
    "vae": VariationalAutoEncoder,
    "flow": NormalizingFlowModel,
}

__all__ = [
    "GenerativeModel",
    "GenerativeAdversarialNetwork",
    "VariationalAutoEncoder",
    "NormalizingFlowModel",
    "MODEL_REGISTRY",
]

