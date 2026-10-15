"""Action values shared by planning and execution, independent of skill services."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleAction:
    op: str
    target: str

def grasp(obj: str) -> SimpleAction:
    return SimpleAction(op="grasp", target=str(obj))

def place(container: str) -> SimpleAction:
    return SimpleAction(op="place", target=str(container))

def observe(obj: str) -> SimpleAction:
    return SimpleAction(op="observe", target=str(obj))
