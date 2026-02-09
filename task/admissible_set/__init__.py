__all__ = ["ASPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "ASPEvaluation":
        from .evaluation import ASPEvaluation
        return ASPEvaluation
    if name == "template_program":
        from .template import template_program
        return template_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
