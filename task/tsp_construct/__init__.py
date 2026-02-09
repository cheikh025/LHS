from .template import template_program

__all__ = ["TSPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "TSPEvaluation":
        from .evaluation import TSPEvaluation
        return TSPEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
