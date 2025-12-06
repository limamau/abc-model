"""The following is a list of abstract classes that are used to define the interface for all models."""

from abc import abstractmethod

from jaxtyping import PyTree
from simple_pytree import Pytree

from .utils import PhysicalConstants


class AbstractState(Pytree):
    pass


class AbstractModel:
    """Abstract model class to define the interface for all models."""


class AbstractRadiationModel(AbstractModel):
    """Abstract radiation model class to define the interface for all radiation models."""

    tstart: float
    """Start time of the model."""

    @abstractmethod
    def run(self, state: PyTree, t: int, dt: float, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError


class AbstractLandModel(AbstractModel):
    """Abstract land model class to define the interface for all land models."""

    # limamau: this could be better coded...
    d1: float

    @abstractmethod
    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: PyTree, dt: float) -> PyTree:
        raise NotImplementedError


class AbstractAtmosphereModel(AbstractModel):
    """Abstract atmosphere model class to define the interface for all atmosphere models."""

    @abstractmethod
    def warmup(self, state: PyTree, const: PhysicalConstants, land) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        state: PyTree,
        const: PhysicalConstants,
    ) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def statistics(self, state: PyTree, t: int, const: PhysicalConstants) -> PyTree:
        raise NotImplementedError

    @abstractmethod
    def integrate(self, state: PyTree, dt: float) -> PyTree:
        raise NotImplementedError
