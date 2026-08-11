"""fractus — unified rebuild of the original systems.

L0: only the native bridge `_core` is exposed (lazily, see below). The modules
nn/, causal/, reasoning/ are added in later layers (L1+).

Design: the pure-Python submodules (fractus.nn, etc.) must remain importable
even if the native Rust module is not built (useful for standalone PyTorch
unit tests). We therefore do NOT import `_core` at module level — we expose it
via lazy __getattr__, which only raises if someone actually accesses it
without having run `maturin develop`.
"""

__version__ = "0.1.0"


def __getattr__(name):
    # Lazy import of the native module fractus._core.
    # Only triggers when someone does `from fractus import _core`
    # or `fractus._core`. The imports `import fractus.nn` do not go through here.
    #
    # We use importlib.import_module rather than `from fractus import _core`,
    # because the latter form would re-trigger __getattr__ → infinite recursion.
    if name == "_core":
        import importlib
        try:
            return importlib.import_module("fractus._core")
        except ImportError as e:
            raise ImportError(
                "The native module fractus._core was not found. "
                "Did you run `maturin develop`?"
            ) from e
    raise AttributeError(f"module 'fractus' has no attribute {name!r}")
