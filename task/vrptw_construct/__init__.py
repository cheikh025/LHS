from .template import template_program

__all__ = ["VRPTWEvaluation", "template_program"]


def __getattr__(name: str):
    if name == "VRPTWEvaluation":
        from .evaluation import VRPTWEvaluation
        return VRPTWEvaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
