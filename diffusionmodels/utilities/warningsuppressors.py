"""
diffusionmodels.utilities.warningsuppressors
============================================

Provides functionalities to suppress various warnings in IDEs.

Functions
---------
unused_variables
    Suppress unused variables by explicit acknowledgements
"""


def unused_variables(*_):
    """
    Suppress unused_variables warnings by declaring the unused_variables.

    Parameters
    ----------
    An arbitrary number of variables that is intentionally unused.
    """
    pass
