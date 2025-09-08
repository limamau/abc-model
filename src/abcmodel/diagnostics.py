from .models import AbstractDiagnostics, AbstractModel


class NoDiagnostics(AbstractDiagnostics[AbstractModel]):
    def __init__(self):
        pass

    def post_init(self, tsteps: int):
        pass

    def store(self, t: int, model: AbstractModel):
        pass
