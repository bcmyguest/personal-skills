"""{{description}}

Mock hello-world package — replace with real code, keeping the shape: logic
lives in importable, typed functions where it's testable; ``main()`` is only
the I/O shim around them.
"""

import sys

__all__ = ["greet"]


def greet(name: str) -> str:
    """Return the greeting for ``name``."""
    return f"Hello, {name}!"


def main() -> None:
    """Console entry point (``[project.scripts]``) — I/O only, no logic.

    Delete this function for a pure library (the lib pyproject.toml has no
    ``[project.scripts]`` table).
    """
    name = sys.argv[1] if len(sys.argv) > 1 else "world"
    print(greet(name))
