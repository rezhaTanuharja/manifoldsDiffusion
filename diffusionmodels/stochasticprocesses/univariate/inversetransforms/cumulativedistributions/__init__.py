"""
Implements various simple `CumulativeDistributionFunction`.

Modules
-------
heatequations       : implements various CDFs from the solution of heat equations
polynomials         : implements various CDFs in the form of polynomials
"""

from . import heatequations, polynomials, uniformso3

__all__ = [
    "heatequations",
    "polynomials",
    "uniformso3",
]
