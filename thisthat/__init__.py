"""this-that-model: typed decisions, one forward pass, no decoding.

    from thisthat import TypedDecider, Question

    decider = TypedDecider.from_pretrained()
    print(decider.decide("temperature 91C, fan stopped",
                         Question("Throttle the process?", ["no", "yes"])))
"""
from .model import DEFAULT_MODEL, TypedDecider, best_device
from .prompt import build, render
from .types import Decision, Question

__version__ = "1.0.0"
__all__ = ["TypedDecider", "Question", "Decision", "build", "render", "DEFAULT_MODEL",
           "best_device", "__version__"]
