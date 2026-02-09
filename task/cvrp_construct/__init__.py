from .template import template_program

__all__ = ["CVRPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "CVRPEvaluation":
        from .evaluation import CVRPEvaluation
        return CVRPEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
