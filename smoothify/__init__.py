"""
Smoothify - Geometry Smoothing Package

A Python package for smoothing and refining geometries derived from raster data
classifications. Transforms jagged polygons and lines resulting from raster-to-vector
conversion into smooth, visually appealing features using an optimized implementation
of Chaikin's corner-cutting algorithm.

Supports:
    - Polygons (including those with holes)
    - LineStrings
    - MultiPolygons
    - MultiLineStrings
    - GeometryCollections
    - GeoDataFrames

Main function:
    smoothify() - Apply Chaikin corner-cutting smoothing to geometries
"""

from .__version__ import __version__
from .coordinator import smoothify

# Package-wide exports
__all__ = [
    "smoothify",
    "__version__",
]
