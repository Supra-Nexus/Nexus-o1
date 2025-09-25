"""
Supra Nexus O1 - Advanced reasoning models with transparent thought
"""

__version__ = "1.0.0"
__author__ = "Supra Foundation LLC"

from supra_nexus.models import SupraModel
from supra_nexus.training import Trainer
from supra_nexus.inference import generate

__all__ = ["SupraModel", "Trainer", "generate"]
