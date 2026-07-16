"""Integration-style test: exercises the public package API from outside,
exactly as a downstream user would. Replace alongside the mock, keeping the
shape — every public function gets a test."""

from {{package_snake}} import greet


def test_greets_by_name() -> None:
    assert greet("Ferris") == "Hello, Ferris!"


def test_greets_world() -> None:
    assert greet("world") == "Hello, world!"
