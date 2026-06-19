from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class EncoderRegistry(Generic[T]):
    """Small registry for pluggable encoders and trackers."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, name: str, builder: Callable[..., T]) -> None:
        if name in self._builders:
            raise KeyError(f"'{name}' is already registered")
        self._builders[name] = builder

    def build(self, name: str, **kwargs) -> T:
        try:
            builder = self._builders[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._builders)) or "<none>"
            raise KeyError(f"Unknown backend '{name}'. Available: {available}") from exc
        return builder(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._builders)
