"""Behaviarium: orchestration/control-plane layer over OpenCV, DeepLabCut, and B-SOiD.

Core is assay-agnostic. The assay (e.g. ``3C_SIT``, ``OFT``) is a config dimension;
stages register through ``behaviarium.registry`` and may be specialised per assay.
"""

__version__ = "0.0.0"
