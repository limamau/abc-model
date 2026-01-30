from typing import Any

import jax.numpy as jnp
from flax import nnx

from .obukhov import ObukhovSurfaceLayerModel


class StabilityEmulator(nnx.Module):
    """Simple MLP to emulate integrated stability functions."""

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input array, typically stability parameter zeta (z/L), shape (..., 1).

        Returns:
            Stability correction value, shape (..., 1).
        """
        x = nnx.Linear(32)(x)
        x = nnx.tanh(x)
        x = nnx.Linear(32)(x)
        x = nnx.tanh(x)
        x = nnx.Linear(1)(x)
        return x


class HybridObukhovSurfaceLayerModel(ObukhovSurfaceLayerModel):
    """Hybrid surface layer model using ML emulators for stability functions.

    This model extends the standard Obukhov model replacing the analytical
    stability functions (Psi_m, Psi_h) with neural networks.
    """

    def __init__(
        self,
        psim_emulator: nnx.Module,
        psih_emulator: nnx.Module,
        psim_params: Any,
        psih_params: Any,
    ):
        """Initialize the hybrid model.

        Args:
            psim_emulator: Flax module for momentum stability function.
            psih_emulator: Flax module for heat stability function.
            psim_params: Parameters for the psim emulator.
            psih_params: Parameters for the psih emulator.
        """
        super().__init__()
        self.psim_emulator = psim_emulator
        self.psih_emulator = psih_emulator
        self.psim_params = psim_params
        self.psih_params = psih_params

    def compute_psim(self, zeta: jnp.ndarray) -> jnp.ndarray:
        """Compute momentum stability function using emulator."""
        original_shape = zeta.shape
        zeta_in = zeta.reshape(-1, 1)
        res = self.psim_emulator.apply(self.psim_params, zeta_in)
        return res.reshape(original_shape)

    def compute_psih(self, zeta: jnp.ndarray) -> jnp.ndarray:
        """Compute heat stability function using emulator."""
        original_shape = zeta.shape
        zeta_in = zeta.reshape(-1, 1)
        res = self.psih_emulator.apply(self.psih_params, zeta_in)
        return res.reshape(original_shape)
