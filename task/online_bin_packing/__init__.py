from .template import template_program

__all__ = ["OBPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "OBPEvaluation":
        from .evaluation import OBPEvaluation
        return OBPEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
