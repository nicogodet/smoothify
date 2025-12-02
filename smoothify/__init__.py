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
    smoothify_with_topology() - Topology-aware smoothing for adjacent polygons
"""

from .__version__ import __version__
from .coordinator import smoothify
from .topology import smoothify_with_topology

# Package-wide exports
__all__ = [
    "smoothify",
    "smoothify_with_topology",
    "__version__",
]
