"""Misc functions for bmctool."""


def truthy_check(value: bool | int | float | str) -> bool:
    """Check if input value is truthy."""
    if isinstance(value, str):
        value = value.lower()
    if value in {True, 1, 1.0, 'true'}:
        return True
    elif value in {False, 0, 'false'}:
        return False
    raise ValueError('Input {value} cannot be converted to bool.')
