"""Misc functions for bmctool."""


def truthy_check(value: bool | int | float | str) -> bool:
    """Check if input value is truthy."""
    if value in {True, 1, 1.0, 'True'}:
        return True
    elif value in {False, 0, 'False'}:
        return False
    raise ValueError('Input {value} cannot be converted to bool.')
