__all__ = ["SCPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "SCPEvaluation":
        from .evaluation import SCPEvaluation
        return SCPEvaluation
    if name == "template_program":
        from .template import template_program
        return template_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
