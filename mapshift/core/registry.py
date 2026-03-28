"""Simple keyed registries used by benchmark components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """A lightweight registry for named benchmark components."""

    name: str
    _items: Dict[str, T] = field(default_factory=dict)

    def register(self, key: str, value: T) -> None:
        if key in self._items:
            raise KeyError(f"{self.name} registry already contains key: {key}")
        self._items[key] = value

    def get(self, key: str) -> T:
        try:
            return self._items[key]
        except KeyError as exc:
            raise KeyError(f"{self.name} registry has no key: {key}") from exc

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def values(self) -> Iterable[T]:
        return self._items.values()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._items.items()

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)
