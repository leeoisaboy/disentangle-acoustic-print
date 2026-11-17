"""Acoustic Print Studio interactive server and UI entrypoints."""

from __future__ import annotations

from typing import Any

__all__ = ["create_fastapi_app", "main"]


def __getattr__(name: str) -> Any:
    if name == "create_fastapi_app":
        from .app import create_fastapi_app

        return create_fastapi_app
    if name == "main":
        from .app import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
