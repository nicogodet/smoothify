from multiprocessing import cpu_count
from typing import Optional, Sequence, overload

import geopandas as gpd
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from smoothify.geometry_ops import (
    _auto_detect_segment_length,
    _smoothify_bulk,
    _smoothify_geodataframe,
    _smoothify_single,
)


@overload
def smoothify(
    geom: gpd.GeoDataFrame,
    segment_length: Optional[float] = None,
    num_cores: int = 0,
    smooth_iterations: int = 3,
    merge_collection: bool = True,
    merge_field: Optional[str] = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    preserve_topology: bool = False,
) -> gpd.GeoDataFrame: ...


@overload
def smoothify(
    geom: Sequence[BaseGeometry],
    segment_length: Optional[float] = None,
    num_cores: int = 0,
    smooth_iterations: int = 3,
    merge_collection: bool = True,
    merge_field: None = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    preserve_topology: bool = False,
) -> BaseGeometry: ...


@overload
def smoothify(
    geom: BaseGeometry,
    segment_length: Optional[float] = None,
    num_cores: int = 0,
    smooth_iterations: int = 3,
    merge_collection: bool = True,
    merge_field: None = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    preserve_topology: bool = False,
) -> BaseGeometry: ...


def smoothify(
    geom: BaseGeometry | Sequence[BaseGeometry] | gpd.GeoDataFrame,
    segment_length: Optional[float] = None,
    num_cores: int = 0,
    smooth_iterations: int = 3,
    merge_collection: bool = True,
    merge_field: Optional[str] = None,
    merge_multipolygons: bool = True,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
    preserve_topology: bool = False,
) -> BaseGeometry | Sequence[BaseGeometry] | gpd.GeoDataFrame:
    """Smooth geometries derived from raster data using Chaikin's corner-cutting algorithm.

    Main entry point for smoothing jagged polygons and lines resulting from
    raster-to-vector conversion. Transforms pixelated edges into smooth, natural-looking
    curves while preserving general shape and area characteristics.

    Supports multiple input types:
    - Single geometries (Polygon, LineString, LinearRing, MultiPolygon, MultiLineString)
    - Lists of geometries or GeometryCollections
    - GeoDataFrames

    Args:
        geom: Geometry, list of geometries, or GeoDataFrame to smooth.
        segment_length: Resolution of the original raster data in map units. Used for
            adding intermediate vertices and simplification. Should match or exceed
            the original raster pixel size. If None (default), automatically detects
            segment length by finding the minimum segment length, which represents the
            true pixel size in pixelated geometries (corners retain minimum length even
            when straight edges are simplified during polygonization).
        num_cores: Number of CPU cores for parallel processing (0 = all available cores,
            1 = serial processing). Only applies to GeoDataFrames and collections.
        smooth_iterations: Number of Chaikin corner-cutting iterations (typically 3-5).
            Higher values produce smoother results but add more vertices.
        merge_collection: Whether to merge/dissolve adjacent geometries in collections
            before smoothing. Useful for joining polygons from adjacent raster cells.
        merge_field: Name of column to use for dissolving geometries in GeoDataFrames.
            Only applies when merge_collection=True.
        merge_multipolygons: Whether to merge adjacent polygons within MultiPolygons
            before smoothing.
        preserve_area: Whether to restore original area after smoothing via buffering.
            Applied to Polygons only.
        area_tolerance: Percentage of original area allowed as error
            (e.g., 0.01 = 0.01% error). Default is 0.01% (99.99% area preservation).
            Smaller values = more accurate area preservation but slower.
            Only affects Polygons when preserve_area=True.
        preserve_topology: Whether to preserve shared boundaries between adjacent
            polygons. When True, uses topology-aware smoothing to ensure no gaps
            or overlaps at shared boundaries. Only applies to GeoDataFrames. Use this
            for datasets where polygons form a complete coverage (like land use maps,
            administrative boundaries, etc.).

    Returns:
        Smoothed geometry matching the input type:
        - BaseGeometry for single geometry inputs
        - list[BaseGeometry] for list inputs
        - GeoDataFrame for GeoDataFrame inputs

    Raises:
        ValueError: If input is not a supported geometry type.

    Examples:
        >>> from smoothify import smoothify
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>>
        >>> # Smooth a single polygon with default area tolerance (0.01%)
        >>> polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        >>> smoothed = smoothify(polygon, segment_length=1.0, smooth_iterations=3)
        >>>
        >>> # Smooth with stricter area preservation (0.001% error)
        >>> smoothed = smoothify(polygon, segment_length=1.0, area_tolerance=0.001)
        >>>
        >>> # Smooth a GeoDataFrame in parallel
        >>> gdf = gpd.read_file("water_bodies.gpkg")
        >>> smoothed_gdf = smoothify(gdf, segment_length=10.0, num_cores=4)
        >>>
        >>> # Smooth adjacent polygons while preserving shared boundaries
        >>> land_use_gdf = gpd.read_file("land_use.gpkg")
        >>> smoothed_gdf = smoothify(land_use_gdf, segment_length=10.0, preserve_topology=True)
    """  # noqa: E501
    if num_cores <= 0:
        num_cores = cpu_count()
    if isinstance(geom, list):
        geom = GeometryCollection(geom)

    if segment_length is None:
        segment_length = _auto_detect_segment_length(geom)

    if merge_field is not None:
        if not isinstance(geom, gpd.GeoDataFrame):
            raise ValueError("merge_field is only supported for GeoDataFrames.")
        if merge_field not in geom.columns:
            raise ValueError(f"merge_field {merge_field} not found in GeoDataFrame.")
        if not merge_collection:
            raise ValueError(
                "merge_field is only supported when merge_collection is True."
            )

    if isinstance(geom, GeometryCollection | MultiPolygon | MultiLineString):
        return _smoothify_bulk(
            geom=geom,
            segment_length=segment_length,
            num_cores=num_cores,
            smooth_iterations=smooth_iterations,
            merge_collection=merge_collection,
            merge_multipolygons=merge_multipolygons,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, Polygon | LineString | LinearRing):
        return _smoothify_single(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            merge_multipolygons=merge_multipolygons,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, gpd.GeoDataFrame):
        return _smoothify_geodataframe(
            gdf=geom,
            segment_length=segment_length,
            num_cores=num_cores,
            smooth_iterations=smooth_iterations,
            merge_collection=merge_collection,
            merge_multipolygons=merge_multipolygons,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
            merge_field=merge_field,
            preserve_topology=preserve_topology,
        )
    else:
        raise ValueError(
            f"Input geometry must be a BaseGeometry or list of BaseGeometry. Got {type(geom)}."  # noqa: E501
        )
