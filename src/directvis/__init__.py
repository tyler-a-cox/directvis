"""Direct-summation visibility simulation for a few point sources, with an fftvis-compatible API."""

from importlib.metadata import PackageNotFoundError, version

from .simulate import get_pos_reds, simulate_vis, simulate_vis_chunks

try:
    __version__ = version("directvis")
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = "unknown"

__all__ = ["simulate_vis", "simulate_vis_chunks", "get_pos_reds", "__version__"]
