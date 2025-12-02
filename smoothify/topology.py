"""Topology-aware smoothing for preserving shared boundaries between adjacent polygons.

This module provides functions for smoothing polygons while preserving the topological
relationships between them. When polygons share boundaries (like land use maps or
administrative boundaries), this ensures that smoothed polygons maintain shared edges
without gaps or overlaps.
"""

from collections import defaultdict

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from .smoothify_core import _chaikin_corner_cutting


def _normalize_edge(p1: tuple, p2: tuple) -> tuple:
    """Normalize an edge so that the smaller point comes first.

    This ensures that edges can be compared regardless of direction.

    Args:
        p1: First endpoint as (x, y) tuple
        p2: Second endpoint as (x, y) tuple

    Returns:
        Tuple of (smaller_point, larger_point) where comparison is lexicographic
    """
    if p1 < p2:
        return (p1, p2)
    return (p2, p1)


def extract_shared_edges(
    geometries: list[Polygon],
) -> tuple[dict[int, LineString], dict[int, list[tuple[int, int]]]]:
    """Extract all unique edges from a collection of polygons.

    Identifies which edges are shared between multiple polygons by comparing
    normalized edge representations.

    Args:
        geometries: List of Polygon geometries to analyze

    Returns:
        A tuple of:
        - edge_dict: Dictionary mapping edge_id -> edge geometry (LineString)
        - polygon_edge_map: Dictionary mapping polygon_id -> list of
          (edge_id, orientation) where orientation is +1 for same direction,
          -1 for reversed
    """
    # Map from normalized edge (as tuple of coords) to edge_id
    edge_to_id: dict[tuple, int] = {}
    # Map from edge_id to LineString geometry
    edge_dict: dict[int, LineString] = {}
    # Map from polygon_id to list of (edge_id, orientation)
    polygon_edge_map: dict[int, list[tuple[int, int]]] = {}

    next_edge_id = 0

    for poly_id, polygon in enumerate(geometries):
        if polygon.is_empty:
            polygon_edge_map[poly_id] = []
            continue

        polygon_edges: list[tuple[int, int]] = []
        coords = list(polygon.exterior.coords)

        for i in range(len(coords) - 1):
            p1 = (coords[i][0], coords[i][1])
            p2 = (coords[i + 1][0], coords[i + 1][1])

            # Skip zero-length edges
            if p1 == p2:
                continue

            normalized = _normalize_edge(p1, p2)

            if normalized in edge_to_id:
                # Edge already exists
                edge_id = edge_to_id[normalized]
                # Determine orientation relative to original edge
                original_edge = edge_dict[edge_id]
                orig_start = (original_edge.coords[0][0], original_edge.coords[0][1])
                orientation = 1 if p1 == orig_start else -1
            else:
                # New edge
                edge_id = next_edge_id
                next_edge_id += 1
                edge_to_id[normalized] = edge_id
                edge_dict[edge_id] = LineString([p1, p2])
                orientation = 1  # Original direction

            polygon_edges.append((edge_id, orientation))

        polygon_edge_map[poly_id] = polygon_edges

    return edge_dict, polygon_edge_map


def _identify_shared_edges(
    polygon_edge_map: dict[int, list[tuple[int, int]]],
) -> set[int]:
    """Identify edges that are shared between multiple polygons.

    Args:
        polygon_edge_map: Dictionary mapping polygon_id -> list of
            (edge_id, orientation)

    Returns:
        Set of edge_ids that are shared by two or more polygons
    """
    edge_usage: dict[int, int] = defaultdict(int)

    for edges in polygon_edge_map.values():
        for edge_id, _ in edges:
            edge_usage[edge_id] += 1

    return {edge_id for edge_id, count in edge_usage.items() if count > 1}


def _build_arcs(
    edge_dict: dict[int, LineString],
    polygon_edge_map: dict[int, list[tuple[int, int]]],
) -> tuple[dict[int, LineString], dict[int, list[tuple[int, int]]]]:
    """Build arcs from edges by merging consecutive edges with same shared status.

    Arcs are maximal sequences of edges that:
    - Are shared by the same set of polygons
    - Form a continuous path (connected endpoints)

    For simplicity, this implementation treats each unique edge as its own arc.
    More sophisticated arc merging could be added later.

    Args:
        edge_dict: Dictionary mapping edge_id -> edge geometry
        polygon_edge_map: Dictionary mapping polygon_id -> list of
            (edge_id, orientation)

    Returns:
        Same format as input (edges = arcs for this simple implementation)
    """
    # For this implementation, we keep edges as individual arcs
    # A more sophisticated version could merge consecutive edges
    return edge_dict, polygon_edge_map


def smooth_edges(
    edges: dict[int, LineString],
    segment_length: float,
    smooth_iterations: int = 3,
) -> dict[int, LineString]:
    """Apply Chaikin smoothing to each unique edge once.

    This ensures that shared edges are smoothed identically when used by
    multiple polygons.

    Args:
        edges: Dictionary mapping edge_id -> edge geometry (LineString)
        segment_length: Target segment length for densification
        smooth_iterations: Number of Chaikin corner-cutting iterations

    Returns:
        Dictionary mapping edge_id -> smoothed edge geometry
    """
    smoothed_edges: dict[int, LineString] = {}

    for edge_id, edge in edges.items():
        if edge.is_empty or len(edge.coords) < 2:
            smoothed_edges[edge_id] = edge
            continue

        # Densify the edge
        segmentized = edge.segmentize(segment_length / 2)

        # Simplify to remove noise
        simplified = segmentized.simplify(
            tolerance=segment_length,
            preserve_topology=True,
        )

        if not isinstance(simplified, LineString) or len(simplified.coords) < 2:
            smoothed_edges[edge_id] = edge
            continue

        # Apply Chaikin smoothing
        # Note: _chaikin_corner_cutting preserves endpoints for LineStrings
        smoothed = _chaikin_corner_cutting(
            geom=simplified,
            num_iterations=smooth_iterations,
        )

        if isinstance(smoothed, LineString):
            smoothed_edges[edge_id] = smoothed
        else:
            smoothed_edges[edge_id] = edge

    return smoothed_edges


def rebuild_polygons_from_edges(
    polygon_edge_map: dict[int, list[tuple[int, int]]],
    smoothed_edges: dict[int, LineString],
    original_geometries: list[Polygon],
) -> list[Polygon]:
    """Reconstruct polygons from smoothed edges, preserving topology.

    Args:
        polygon_edge_map: Dictionary mapping polygon_id -> list of
            (edge_id, orientation)
        smoothed_edges: Dictionary mapping edge_id -> smoothed edge geometry
        original_geometries: Original polygon list for fallback

    Returns:
        List of reconstructed polygons in the same order as original_geometries
    """
    result_polygons: list[Polygon] = []

    for poly_id in range(len(original_geometries)):
        if poly_id not in polygon_edge_map or not polygon_edge_map[poly_id]:
            # Keep original if no edges mapped
            result_polygons.append(original_geometries[poly_id])
            continue

        # Collect all coordinates from edges in order
        all_coords: list[tuple[float, float]] = []

        for edge_id, orientation in polygon_edge_map[poly_id]:
            if edge_id not in smoothed_edges:
                continue

            edge = smoothed_edges[edge_id]
            coords = list(edge.coords)

            if orientation == -1:
                # Reverse the edge
                coords = coords[::-1]

            if not all_coords:
                # First edge - add all coordinates
                all_coords.extend(coords)
            else:
                # Subsequent edges - skip first point (it should match last)
                all_coords.extend(coords[1:])

        # Close the ring if needed
        if len(all_coords) >= 3:
            if all_coords[0] != all_coords[-1]:
                all_coords.append(all_coords[0])

            try:
                polygon = Polygon(all_coords)
                if polygon.is_valid:
                    result_polygons.append(polygon)
                else:
                    # Try to fix with make_valid
                    from shapely import make_valid

                    fixed = make_valid(polygon)
                    if isinstance(fixed, Polygon):
                        result_polygons.append(fixed)
                    elif isinstance(fixed, MultiPolygon):
                        # Take the largest polygon
                        largest = max(fixed.geoms, key=lambda x: x.area)
                        result_polygons.append(largest)
                    else:
                        result_polygons.append(original_geometries[poly_id])
            except Exception:
                result_polygons.append(original_geometries[poly_id])
        else:
            result_polygons.append(original_geometries[poly_id])

    return result_polygons


def _apply_area_preservation(
    smoothed_polygons: list[Polygon],
    original_polygons: list[Polygon],
    area_tolerance: float,
) -> list[Polygon]:
    """Apply area preservation to smoothed polygons.

    Args:
        smoothed_polygons: List of smoothed polygons
        original_polygons: List of original polygons (for target areas)
        area_tolerance: Percentage of original area allowed as error

    Returns:
        List of area-preserved polygons
    """
    from .smoothify_core import _preserve_area_with_buffer

    result = []
    for smoothed, original in zip(smoothed_polygons, original_polygons, strict=True):
        if original.is_empty or smoothed.is_empty:
            result.append(smoothed)
            continue

        target_area = original.area
        absolute_tolerance = target_area * (area_tolerance / 100.0)

        preserved = _preserve_area_with_buffer(
            polygon=smoothed,
            target_area=target_area,
            tolerance=absolute_tolerance,
        )
        result.append(preserved)

    return result


def smoothify_with_topology(
    geometries: list[Polygon] | gpd.GeoDataFrame,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
) -> list[Polygon] | gpd.GeoDataFrame:
    """Apply topology-aware smoothing to a collection of polygons.

    This function preserves shared boundaries between adjacent polygons by:
    1. Extracting all unique edges from the polygons
    2. Smoothing each unique edge exactly once
    3. Rebuilding polygons from the smoothed edges

    This ensures that adjacent polygons maintain shared boundaries without
    gaps or overlaps after smoothing.

    Args:
        geometries: List of Polygon geometries or a GeoDataFrame to smooth
        segment_length: Target segment length for densification (typically
            raster pixel size)
        smooth_iterations: Number of Chaikin corner-cutting iterations (default: 3)
        preserve_area: Whether to restore original area after smoothing (default: True)
        area_tolerance: Percentage of original area allowed as error (default: 0.01%)

    Returns:
        Smoothed geometries in the same format as input:
        - List of Polygons if input was list
        - GeoDataFrame if input was GeoDataFrame

    Example:
        >>> from smoothify.topology import smoothify_with_topology
        >>> from shapely.geometry import Polygon
        >>>
        >>> # Two adjacent squares
        >>> poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        >>>
        >>> smoothed = smoothify_with_topology([poly1, poly2], segment_length=1.0)
        >>> # Shared edge at x=10 is smoothed identically for both polygons
    """
    # Handle GeoDataFrame input
    is_geodataframe = isinstance(geometries, gpd.GeoDataFrame)
    if is_geodataframe:
        gdf = geometries
        polygon_list = list(gdf.geometry)
    else:
        polygon_list = list(geometries)

    # Filter to only process Polygon geometries
    polygon_indices: list[int] = []
    polygons_to_process: list[Polygon] = []
    other_geometries: dict[int, BaseGeometry] = {}

    for i, geom in enumerate(polygon_list):
        if isinstance(geom, Polygon) and not geom.is_empty:
            polygon_indices.append(i)
            polygons_to_process.append(geom)
        else:
            other_geometries[i] = geom

    if not polygons_to_process:
        # No polygons to process
        return geometries

    # Extract edges
    edge_dict, polygon_edge_map = extract_shared_edges(polygons_to_process)

    # Build arcs (for now, edges = arcs)
    arc_dict, polygon_arc_map = _build_arcs(edge_dict, polygon_edge_map)

    # Smooth edges
    smoothed_edges = smooth_edges(
        edges=arc_dict,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
    )

    # Rebuild polygons
    smoothed_polygons = rebuild_polygons_from_edges(
        polygon_edge_map=polygon_arc_map,
        smoothed_edges=smoothed_edges,
        original_geometries=polygons_to_process,
    )

    # Apply area preservation if requested
    if preserve_area:
        smoothed_polygons = _apply_area_preservation(
            smoothed_polygons=smoothed_polygons,
            original_polygons=polygons_to_process,
            area_tolerance=area_tolerance,
        )

    # Reconstruct full result with non-polygon geometries in original positions
    result_list: list[BaseGeometry] = [None] * len(polygon_list)  # type: ignore

    # Place smoothed polygons back
    for orig_idx, smoothed_poly in zip(polygon_indices, smoothed_polygons, strict=True):
        result_list[orig_idx] = smoothed_poly

    # Place non-polygon geometries back
    for idx, geom in other_geometries.items():
        result_list[idx] = geom

    # Return in appropriate format
    if is_geodataframe:
        result_gdf = gdf.copy()
        result_gdf.geometry = result_list
        return result_gdf
    else:
        return [g for g in result_list if isinstance(g, Polygon)]
