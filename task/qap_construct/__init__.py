__all__ = ["QAPEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "QAPEvaluation":
        from .evaluation import QAPEvaluation
        return QAPEvaluation
    if name == "template_program":
        from .template import template_program
        return template_program
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
