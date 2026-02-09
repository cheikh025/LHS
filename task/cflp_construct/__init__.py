__all__ = ["CFLPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "CFLPEvaluation":
        from .evaluation import CFLPEvaluation
        return CFLPEvaluation
    if name == "template_program":
        from .template import template_program
        return template_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
