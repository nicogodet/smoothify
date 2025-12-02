from functools import partial
from multiprocessing import get_context
from typing import Optional, Sequence, cast

import geopandas as gpd
import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from .smoothify_core import _join_adjacent, _smoothify_geometry


def _smoothify_multipolygon(
    geom: BaseGeometry,
    segment_length: float,
    smooth_iterations: int,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> MultiPolygon:
    """Smooth a MultiPolygon, optionally merging adjacent polygons first.

    If merge_multipolygons is True, applies a small buffer-dissolve operation
    to join adjacent polygons before smoothing. This is useful for MultiPolygons
    derived from adjacent raster cells with the same classification."""

    if merge_multipolygons:
        geom = _join_adjacent(geom=geom, segment_length=segment_length)

    if isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)

    elif isinstance(geom, Polygon):
        polygons = [geom]

    else:
        raise ValueError(
            f"Expected output of unary_union to be MultiPolygon or Polygon, got {type(geom)}"  # noqa: E501
        )
    for polygon in polygons:
        assert isinstance(polygon, Polygon)

    polygons = [
        _smoothify_polygon(
            geom=polygon,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
        for polygon in polygons
    ]
    return MultiPolygon(polygons)


def _smoothify_linearing(
    geom: LinearRing,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> LinearRing:
    """Smooth a LinearRing by converting to Polygon, smoothing, then extracting the ring.

    LinearRings are closed loops that can be smoothed using polygon smoothing
    techniques, then converted back to a LinearRing. Uses default area_tolerance
    of 0.01% for area preservation when preserve_area is True."""  # noqa: E501

    smoothed_polygon = _smoothify_geometry(
        geom=Polygon(geom),
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
    )
    # Cast to Polygon since we passed in a Polygon
    smoothed_polygon = cast(Polygon, smoothed_polygon)
    return LinearRing(smoothed_polygon.exterior.coords)


def _extract_and_fill_holes(geom: Polygon) -> tuple[list[Polygon], Polygon]:
    """Extract interior holes from a polygon as separate polygons.

    Separates a polygon's interior rings (holes) from its exterior shell so they
    can be smoothed independently. This enables individual smoothing of holes and
    maintains hole area."""  # noqa: E501

    holes = []
    for interior in geom.interiors:
        holes.append(Polygon(interior))
    filled_polygon = Polygon(geom.exterior)
    return holes, filled_polygon


def _smoothify_polygon(
    geom: Polygon,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
) -> Polygon:
    """Smooth a Polygon while preserving interior holes.

    Smooths the exterior shell and each interior hole independently, then
    recombines them. This approach prevents artifacts at hole boundaries and
    maintains proper polygon topology."""

    holes, filled_polygon = _extract_and_fill_holes(geom)

    smooth_polygon = _smoothify_geometry(
        geom=filled_polygon,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )

    # Smooth all holes and subtract them from the smoothed exterior
    if holes and isinstance(smooth_polygon, Polygon):
        smoothed_hole_polygons = []
        for hole in holes:
            smooth_hole = _smoothify_geometry(
                geom=hole,
                segment_length=segment_length,
                smooth_iterations=smooth_iterations,
                preserve_area=preserve_area,
                area_tolerance=area_tolerance,
            )
            if isinstance(smooth_hole, Polygon):
                smoothed_hole_polygons.append(smooth_hole)

        # Use difference to subtract holes, ensuring they don't add area
        # even if they extend outside the smoothed exterior
        if smoothed_hole_polygons:
            # Intersect each hole with the smooth_polygon first to ensure
            # we only subtract the portion that's actually inside
            for hole_poly in smoothed_hole_polygons:
                # Only subtract the part of the hole that's inside the polygon
                hole_inside = smooth_polygon.intersection(hole_poly)
                if not hole_inside.is_empty:
                    smooth_polygon = smooth_polygon.difference(hole_inside)

    if not isinstance(smooth_polygon, Polygon):
        raise ValueError(
            f"Expected output of smoothify_geometry to be Polygon, got {type(smooth_polygon)}"  # noqa: E501
        )
    return smooth_polygon


def _smoothify_multilinestring(
    geom: MultiLineString,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> MultiLineString:
    """Smooth every LineString within a MultiLineString independently.

    Applies smoothing to each line segment in the collection while maintaining
    the MultiLineString structure. Area preservation is not applicable to
    LineStrings and is ignored."""

    lines = [
        _smoothify_linestring(
            geom=line,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
        )
        for line in geom.geoms
    ]
    return MultiLineString(lines)


def _smoothify_linestring(
    geom: LineString,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> LineString:
    """Smooth a LineString using Chaikin's corner-cutting algorithm.

    Transforms jagged line features (e.g., road networks from classified imagery)
    into smooth curves while preserving endpoints."""

    smooth_line = _smoothify_geometry(
        geom=geom,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
    )
    if not isinstance(smooth_line, LineString):
        raise ValueError(
            f"Expected output of smoothify_geometry to be LineString, got {type(smooth_line)}"  # noqa: E501
        )
    return smooth_line


def _smoothify_geodataframe(
    gdf: gpd.GeoDataFrame,
    segment_length: float,
    num_cores: int,
    smooth_iterations: int,
    merge_collection: bool,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
    merge_field: Optional[str] = None,
    preserve_topology: bool = False,
) -> gpd.GeoDataFrame:
    """Smooth all geometries in a GeoDataFrame with optional parallel processing.

    Processes each geometry in a GeoDataFrame using Chaikin corner cutting.
    Optionally merges adjacent features before smoothing and supports parallel
    execution for large datasets.

    Args:
        gdf: GeoDataFrame to smooth
        segment_length: Target segment length for densification
        num_cores: Number of CPU cores for parallel processing
        smooth_iterations: Number of Chaikin corner-cutting iterations
        merge_collection: Whether to merge adjacent polygons before smoothing
        merge_multipolygons: Whether to merge adjacent polygons within MultiPolygons
        preserve_area: Whether to restore original area after smoothing
        area_tolerance: Percentage of original area allowed as error
        merge_field: Column name for grouping polygons during dissolve
        preserve_topology: Whether to preserve shared boundaries between adjacent
            polygons. When True, uses topology-aware smoothing to ensure no gaps
            or overlaps at shared boundaries.
    """  # noqa: E501

    modified_gdf = gdf.copy()

    # Use topology-aware smoothing if requested
    if preserve_topology:
        from .topology import smoothify_with_topology

        return smoothify_with_topology(
            geometries=modified_gdf,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )

    if merge_collection:
        # Only merge polygons, not linestrings
        # Separate polygons from other geometry types
        polygon_mask = modified_gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])

        if polygon_mask.any():
            # Process polygons
            polygon_gdf = modified_gdf[polygon_mask].copy()
            polygon_gdf.geometry = polygon_gdf.buffer(segment_length / 1000)
            polygon_gdf = polygon_gdf.dissolve(by=merge_field)
            if merge_field is not None:
                polygon_gdf = polygon_gdf.reset_index()
            polygon_gdf = polygon_gdf.explode(index_parts=False, ignore_index=True)

            # Combine with non-polygon geometries
            if not polygon_mask.all():
                other_gdf = modified_gdf[~polygon_mask].copy()
                modified_gdf = gpd.GeoDataFrame(
                    pd.concat([polygon_gdf, other_gdf], ignore_index=True),
                    crs=gdf.crs
                )
            else:
                modified_gdf = polygon_gdf
        # If no polygons, leave modified_gdf as is

    smoothify_partial = partial(
        _smoothify_single,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )

    if num_cores == 1:
        modified_gdf.geometry = [
            smoothify_partial(geom) for geom in modified_gdf.geometry
        ]

    else:
        modified_gdf.geometry = Parallel(n_jobs=num_cores)(
            delayed(smoothify_partial)(
                geom,  # type: ignore
            )
            for geom in modified_gdf.geometry
        )
    return modified_gdf


def _smoothify_single(
    geom: BaseGeometry,
    segment_length: float,
    smooth_iterations: int,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> BaseGeometry:
    """Smooth a single geometry by dispatching to the appropriate type-specific function.

    Central dispatcher that routes geometries to specialized smoothing functions
    based on their type. Handles all supported geometry types including simple
    and multi-part geometries."""  # noqa: E501
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return _smoothify_polygon(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, LinearRing):
        return _smoothify_linearing(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    elif isinstance(geom, LineString):
        return _smoothify_linestring(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    elif isinstance(geom, MultiPolygon):
        return _smoothify_multipolygon(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            merge_multipolygons=merge_multipolygons,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, MultiLineString):
        return _smoothify_multilinestring(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    return geom


def _smoothify_bulk(
    geom: GeometryCollection | MultiPolygon | MultiLineString,
    segment_length: float,
    num_cores: int,
    smooth_iterations: int,
    merge_collection: bool,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> GeometryCollection | MultiPolygon | MultiLineString:
    """Smooth a collection of geometries using parallel processing.

    Processes GeometryCollection, MultiPolygon, or MultiLineString by distributing
    component geometries across multiple worker processes for efficient parallel
    execution on large datasets.

    Optionally merges adjacent geometries before smoothing, which is useful for
    collections of polygons derived from adjacent raster cells with the same
    classification."""

    input_type = type(geom)

    # join adjacent polygons found in a geometry collection
    if merge_collection and isinstance(geom, GeometryCollection):
        geom_joined = _join_adjacent(geom=geom, segment_length=segment_length)
        geom = GeometryCollection(geom_joined)

    # join adjacent polygons found within a multipolygon
    if merge_multipolygons and isinstance(geom, MultiPolygon):
        geom_joined = _join_adjacent(geom=geom, segment_length=segment_length)
        geom = GeometryCollection(geom_joined)

    if isinstance(geom, MultiLineString):
        geom = GeometryCollection(geom.geoms)

    smoothify_partial = partial(
        _smoothify_single,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )
    if num_cores == 1:
        geom_smoothed = list(map(smoothify_partial, geom.geoms))
    else:
        # Use spawn to avoid fork warnings in multi-threaded environments (Python 3.13+)
        ctx = get_context("spawn")
        with ctx.Pool(num_cores) as pool:
            geom_smoothed = pool.map(smoothify_partial, geom.geoms)

    if input_type == MultiPolygon:
        if geom_smoothed and all(isinstance(g, Polygon) for g in geom_smoothed):
            return MultiPolygon(cast(list[Polygon], geom_smoothed))
    elif input_type == MultiLineString:
        if geom_smoothed and all(isinstance(g, LineString) for g in geom_smoothed):
            return MultiLineString(cast(list[LineString], geom_smoothed))

    return GeometryCollection(geom_smoothed)


def _auto_detect_segment_length(
    geom: BaseGeometry | Sequence[BaseGeometry] | gpd.GeoDataFrame,
) -> float:
    """Auto-detect segment length from the minimum segment length of the geometry.

    For pixelated/rasterized geometries, finds the shortest segment which typically
    represents the true pixel size. Polygonization often skips vertices along straight
    edges, but corners retain the minimum segment length equal to the pixel size.

    Samples up to 100 segments per geometry for efficiency on large datasets.

    Args:
        geom: Geometry, Sequence of geometries, GeoDataFrame, or collection to analyze.

    Returns:
        float: Estimated segment length based on minimum segment length.

    Raises:
        ValueError: If no suitable geometry found for auto-detection.
    """

    def get_min_segment_length(
        geom: BaseGeometry, max_samples: int = 100
    ) -> Optional[float]:
        """Get the minimum segment length from a geometry.

        Samples up to max_samples segments to find the shortest one.
        For pixelated geometries, the minimum segment represents the pixel size.
        """
        import numpy as np

        if geom.is_empty:
            return None

        if isinstance(geom, Point):
            return None

        min_length = None

        def compute_min_segment_from_coords(coords_list):
            """Vectorized computation of minimum segment length from coordinates."""
            coords_array = np.array(coords_list)
            if len(coords_array) < 2:
                return None

            # Vectorized distance calculation
            diffs = coords_array[1:] - coords_array[:-1]
            distances = np.linalg.norm(diffs, axis=1)

            # Filter out zero-length segments
            valid_distances = distances[distances > 0]
            if len(valid_distances) == 0:
                return None

            # Sample if too many segments
            if len(valid_distances) > max_samples:
                step = len(valid_distances) // max_samples
                valid_distances = valid_distances[::step]

            return float(np.min(valid_distances))

        if isinstance(geom, Polygon):
            # Check exterior ring
            coords = list(geom.exterior.coords)
            length = compute_min_segment_from_coords(coords)
            if length is not None:
                min_length = length

            # Check interior rings (holes)
            for interior in geom.interiors:
                coords = list(interior.coords)
                length = compute_min_segment_from_coords(coords)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        elif isinstance(geom, (LineString, LinearRing)):
            coords = list(geom.coords)
            min_length = compute_min_segment_from_coords(coords)

        elif isinstance(geom, (MultiPolygon, MultiLineString)):
            for sub_geom in geom.geoms:
                length = get_min_segment_length(sub_geom, max_samples)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        elif isinstance(geom, GeometryCollection):
            for sub_geom in geom.geoms:
                length = get_min_segment_length(sub_geom, max_samples)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        return min_length

    # Handle GeoDataFrame
    min_segment_length = None
    if isinstance(geom, gpd.GeoDataFrame):
        # Sample first few geometries to find minimum
        for geometry in geom.geometry.head(10):
            length = get_min_segment_length(geometry)
            if length is not None:
                if min_segment_length is None or length < min_segment_length:
                    min_segment_length = length
    elif isinstance(geom, Sequence) and not isinstance(geom, (str, bytes)):
        # Handle sequence of geometries
        for i, geometry in enumerate(geom):
            if i >= 10:  # Sample first 10 geometries
                break
            if isinstance(geometry, BaseGeometry):
                length = get_min_segment_length(geometry)
                if length is not None:
                    if min_segment_length is None or length < min_segment_length:
                        min_segment_length = length
    else:
        # Handle single geometry or collection
        min_segment_length = get_min_segment_length(cast(BaseGeometry, geom))

    if min_segment_length is not None:
        return min_segment_length

    raise ValueError(
        "Could not auto-detect segment_length: no suitable geometry with coordinates found. "  # noqa: E501
        "Please provide segment_length explicitly."
    )
