"""Provides functionalities to suppress various warnings by static type checker.

Functions
---------
`unused_variables`    : A function to explicitly declare that variables are not used
"""


def unused_variables(*_) -> None:
    """Suppress `unused_variables` warnings by explicit declaration.

    Parameters
    ----------
    An arbitrary number of variables that is intentionally unused

    Usage Example
    -------------
    ```python
    class ChildClass(ParentClass):

        def __init__(self, argument_a: Any, argument_b: Any):
            unused_variables(argument_a, argument_b)
            pass
    ```
    """
    pass
