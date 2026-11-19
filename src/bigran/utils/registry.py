"""Minimal model registry."""
from __future__ import annotations

from typing import Any, Callable, Dict


class Registry:
    def __init__(self, what: str) -> None:
        self.what = what
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, obj: Callable[..., Any] | None = None):
        if obj is not None:
            self._items[name] = obj
            return obj

        def deco(fn):
            self._items[name] = fn
            return fn

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            raise KeyError(f"未知的 {self.what}: {name!r}；可用: {sorted(self._items)}")
        return self._items[name]

    def build(self, name: str, **kwargs) -> Any:
        return self.get(name)(**kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items)

    def names(self) -> list[str]:
        return sorted(self._items)


MODELS = Registry("模型")