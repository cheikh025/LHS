from .template import template_program

__all__ = ["JSSPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "JSSPEvaluation":
        from .evaluation import JSSPEvaluation
        return JSSPEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
