"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod

from jaxtyping import PyTree

from .utils import PhysicalConstants


class AbstractState:
    """Abstract state class to define the interface for all states."""


class AbstractRadiationState(AbstractState):
    """Abstract radiation state."""


class AbstractLandState(AbstractState):
    """Abstract land state."""


class AbstractAtmosphereState(AbstractState):
    """Abstract atmosphere state."""


class AbstractCoupledState(AbstractState):
    """Abstract coupled state."""
    atmosphere: AbstractAtmosphereState
    land: AbstractLandState
    radiation: AbstractRadiationState


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel):
    """Abstract radiation model class to define the interface for all radiation models."""

    tstart: float
    """Start time of the model."""

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState,
        t: int,
        dt: float,
        const: PhysicalConstants
    ) -> AbstractRadiationState:
        raise NotImplementedError


class AbstractLandModel(AbstractModel):
    """Abstract land model class to define the interface for all land models."""

    # limamau: this could be better coded...
    d1: float

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> AbstractLandState:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AbstractLandState, dt: float) -> AbstractLandState:
        raise NotImplementedError


class AbstractAtmosphereModel(AbstractModel):
    """Abstract atmosphere model class to define the interface for all atmosphere models."""

    @abstractmethod
    def warmup(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
        land: AbstractLandModel
    ) -> AbstractCoupledState:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: AbstractCoupledState,
        const: PhysicalConstants,
    ) -> AbstractAtmosphereState:
        raise NotImplementedError

    @abstractmethod
    def statistics(
        self,
        state: AbstractCoupledState,
        t: int,
        const: PhysicalConstants
    ) -> AbstractCoupledState:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: AbstractAtmosphereState, dt: float) -> AbstractAtmosphereState:
        raise NotImplementedError
