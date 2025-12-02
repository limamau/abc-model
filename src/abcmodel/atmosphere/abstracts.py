"""Abstract classes for atmosphere sub-modules."""

from abc import abstractmethod

from jaxtyping import Array, PyTree

from ..abstracts import AbstractCoupledState, AbstractModel, AbstractState
from ..utils import PhysicalConstants


class AbstractSurfaceLayerState(AbstractState):
    """Abstract surface layer state."""


class AbstractMixedLayerState(AbstractState):
    """Abstract mixed layer state."""


class AbstractCloudState(AbstractState):
    """Abstract cloud state."""


class AbstractSurfaceLayerModel(AbstractModel):
    """Abstract surface layer model class to define the interface for all surface layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> AbstractSurfaceLayerState:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_ra(state: AbstractSurfaceLayerState) -> Array:
        raise NotImplementedError


class AbstractMixedLayerModel(AbstractModel):
    """Abstract mixed layer model class to define the interface for all mixed layer models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> AbstractMixedLayerState:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, state: AbstractMixedLayerState, t: int, const: PhysicalConstants) -> AbstractMixedLayerState:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AbstractMixedLayerState, dt: float) -> AbstractMixedLayerState:
        raise NotImplementedError


class AbstractCloudModel(AbstractModel):
    """Abstract cloud model class to define the interface for all cloud models."""

    @abstractmethod
    def run(self, state: AbstractCoupledState, const: PhysicalConstants) -> AbstractCloudState:
        raise NotImplementedError
