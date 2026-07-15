"""Jinja2 template loading for all markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

_env = Environment(
    loader=FileSystemLoader(Path(__file__).parent),
    keep_trailing_newline=True,
)


def render(template_name: str, **context: Any) -> str:
    return _env.get_template(template_name).render(**context)
