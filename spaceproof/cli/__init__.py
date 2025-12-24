"""SpaceProof CLI package."""

from .args import create_parser
from .dispatch import dispatch

__all__ = ["create_parser", "dispatch"]
