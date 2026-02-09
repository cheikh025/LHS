from .template import template_program

__all__ = ["KnapsackEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "KnapsackEvaluation":
        from .evaluation import KnapsackEvaluation
        return KnapsackEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
