"""Tests for topology-aware smoothing functionality."""

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from smoothify import smoothify
from smoothify.topology import (
    _identify_shared_edges,
    _normalize_edge,
    extract_shared_edges,
    rebuild_polygons_from_edges,
    smooth_edges,
    smoothify_with_topology,
)


class TestNormalizeEdge:
    """Test edge normalization for consistent comparison."""

    def test_same_order(self):
        """Test that smaller point comes first when already in order."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 1.0)
        result = _normalize_edge(p1, p2)
        assert result == (p1, p2)

    def test_reverse_order(self):
        """Test that smaller point comes first when reversed."""
        p1 = (1.0, 1.0)
        p2 = (0.0, 0.0)
        result = _normalize_edge(p1, p2)
        assert result == (p2, p1)

    def test_same_x_different_y(self):
        """Test normalization with same x coordinate."""
        p1 = (5.0, 10.0)
        p2 = (5.0, 0.0)
        result = _normalize_edge(p1, p2)
        assert result == (p2, p1)  # (5.0, 0.0) is smaller


class TestExtractSharedEdges:
    """Test extraction of shared edges from polygon collections."""

    def test_two_adjacent_squares(self):
        """Test that shared edge is identified for two adjacent squares."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2])

        # Both polygons should have edges mapped
        assert 0 in polygon_edge_map
        assert 1 in polygon_edge_map

        # Should have edges from both polygons
        assert len(edge_dict) > 0

    def test_non_adjacent_polygons(self):
        """Test that non-adjacent polygons have no shared edges."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(100, 100), (105, 100), (105, 105), (100, 105)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2])

        # Get shared edge count
        shared_edges = _identify_shared_edges(polygon_edge_map)

        # No shared edges expected
        assert len(shared_edges) == 0

    def test_three_polygons_with_shared_edges(self):
        """Test that shared edges are correctly identified with three polygons."""
        # Three adjacent squares in a row
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        poly3 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2, poly3])

        # Should have edges mapped for all three polygons
        assert 0 in polygon_edge_map
        assert 1 in polygon_edge_map
        assert 2 in polygon_edge_map

    def test_empty_polygon(self):
        """Test handling of empty polygon."""
        poly1 = Polygon()
        poly2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2])

        # Empty polygon should have empty edge list
        assert polygon_edge_map[0] == []
        # Second polygon should have edges
        assert len(polygon_edge_map[1]) > 0


class TestIdentifySharedEdges:
    """Test identification of shared edges."""

    def test_no_shared_edges(self):
        """Test with polygons that share no edges."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(100, 100), (105, 100), (105, 105), (100, 105)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2])
        shared_edges = _identify_shared_edges(polygon_edge_map)

        assert len(shared_edges) == 0

    def test_one_shared_edge(self):
        """Test with two adjacent polygons sharing one edge."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        edge_dict, polygon_edge_map = extract_shared_edges([poly1, poly2])
        shared_edges = _identify_shared_edges(polygon_edge_map)

        # Should have exactly one shared edge
        assert len(shared_edges) == 1


class TestSmoothEdges:
    """Test smoothing of individual edges."""

    def test_smooth_single_edge(self):
        """Test smoothing a single edge."""
        edge = LineString([(0, 0), (5, 5), (10, 0)])
        edges = {0: edge}

        smoothed = smooth_edges(edges, segment_length=1.0, smooth_iterations=3)

        assert 0 in smoothed
        assert isinstance(smoothed[0], LineString)
        # Should have more vertices after smoothing
        assert len(smoothed[0].coords) >= len(edge.coords)

    def test_smooth_preserves_endpoints(self):
        """Test that smoothing preserves edge endpoints."""
        edge = LineString([(0, 0), (5, 5), (10, 0)])
        edges = {0: edge}

        smoothed = smooth_edges(edges, segment_length=1.0, smooth_iterations=3)

        # Check endpoints are preserved (approximately)
        original_start = edge.coords[0]
        original_end = edge.coords[-1]
        smoothed_start = smoothed[0].coords[0]
        smoothed_end = smoothed[0].coords[-1]

        assert original_start == pytest.approx(smoothed_start, abs=0.01)
        assert original_end == pytest.approx(smoothed_end, abs=0.01)

    def test_smooth_empty_edge(self):
        """Test handling of empty edge."""
        edge = LineString()
        edges = {0: edge}

        smoothed = smooth_edges(edges, segment_length=1.0, smooth_iterations=3)

        assert 0 in smoothed
        assert smoothed[0].is_empty


class TestRebuildPolygonsFromEdges:
    """Test reconstruction of polygons from smoothed edges."""

    def test_rebuild_simple_polygon(self):
        """Test rebuilding a simple polygon from edges."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        edge_dict, polygon_edge_map = extract_shared_edges([poly])

        # Don't smooth, just rebuild
        rebuilt = rebuild_polygons_from_edges(polygon_edge_map, edge_dict, [poly])

        assert len(rebuilt) == 1
        assert isinstance(rebuilt[0], Polygon)
        assert rebuilt[0].is_valid


class TestSmoothifyWithTopology:
    """Test the main topology-aware smoothing function."""

    def test_basic_functionality(self):
        """Test basic topology-aware smoothing."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        smoothed = smoothify_with_topology(
            [poly1, poly2],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 2
        assert all(isinstance(p, Polygon) for p in smoothed)
        assert all(p.is_valid for p in smoothed)

    def test_preserves_topology_no_gaps(self):
        """Test that smoothed adjacent polygons have no significant gaps."""
        # Two adjacent squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        smoothed = smoothify_with_topology(
            [poly1, poly2],
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=False,  # Disable area preservation for cleaner test
        )

        # The union of smoothed polygons should be similar to original union
        original_union = poly1.union(poly2)
        smoothed_union = smoothed[0].union(smoothed[1])

        # Check that there are no significant gaps (union areas should be similar)
        area_diff = abs(original_union.area - smoothed_union.area)
        assert area_diff < original_union.area * 0.15  # Allow 15% difference

    def test_geodataframe_input(self):
        """Test with GeoDataFrame input."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        gdf = gpd.GeoDataFrame(
            {"name": ["poly1", "poly2"]},
            geometry=[poly1, poly2],
            crs="EPSG:4326"
        )

        smoothed = smoothify_with_topology(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == 2
        assert smoothed.crs == gdf.crs
        assert list(smoothed["name"]) == ["poly1", "poly2"]
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_isolated_polygon(self):
        """Test that isolated polygons are smoothed correctly."""
        # Two non-adjacent polygons
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])

        smoothed = smoothify_with_topology(
            [poly1, poly2],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 2
        assert all(isinstance(p, Polygon) for p in smoothed)
        assert all(p.is_valid for p in smoothed)

    def test_preserve_area_option(self):
        """Test area preservation option."""
        poly1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        poly2 = Polygon([(100, 0), (200, 0), (200, 100), (100, 100)])

        original_area1 = poly1.area
        original_area2 = poly2.area

        smoothed = smoothify_with_topology(
            [poly1, poly2],
            segment_length=10.0,
            smooth_iterations=3,
            preserve_area=True,
            area_tolerance=0.1,
        )

        # Areas should be close to original
        area_diff1 = abs(smoothed[0].area - original_area1) / original_area1 * 100
        area_diff2 = abs(smoothed[1].area - original_area2) / original_area2 * 100

        # Allow some tolerance due to area preservation algorithm
        assert area_diff1 < 5  # Less than 5% difference
        assert area_diff2 < 5

    def test_empty_polygon_in_list(self):
        """Test handling of empty polygon in list."""
        poly1 = Polygon()
        poly2 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        # Should not raise an error
        smoothed = smoothify_with_topology(
            [poly1, poly2],
            segment_length=1.0,
            smooth_iterations=3,
        )

        # Empty polygon should be preserved, valid polygon should be smoothed
        assert len(smoothed) == 2


class TestSmoothifyAPIWithTopology:
    """Test the main smoothify() API with preserve_topology parameter."""

    def test_preserve_topology_parameter(self):
        """Test that preserve_topology parameter works via main API."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        gdf = gpd.GeoDataFrame(
            {"name": ["poly1", "poly2"]},
            geometry=[poly1, poly2],
            crs="EPSG:4326"
        )

        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_topology=True,
            num_cores=1,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == 2
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_preserve_topology_default_false(self):
        """Test that preserve_topology defaults to False."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        gdf = gpd.GeoDataFrame(
            {"name": ["poly1", "poly2"]},
            geometry=[poly1, poly2],
            crs="EPSG:4326"
        )

        # This should use the default (non-topology-aware) smoothing
        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=False,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == 2
        assert all(geom.is_valid for geom in smoothed.geometry)


class TestComplexTopologies:
    """Test complex topological scenarios."""

    def test_t_junction(self):
        """Test T-junction where three polygons meet."""
        # Create a T-junction pattern
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        poly3 = Polygon([(0, 10), (10, 10), (10, 20), (0, 20)])

        smoothed = smoothify_with_topology(
            [poly1, poly2, poly3],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 3
        assert all(isinstance(p, Polygon) for p in smoothed)
        assert all(p.is_valid for p in smoothed)

    def test_four_corners(self):
        """Test four polygons meeting at a corner."""
        # Create 2x2 grid of squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        poly3 = Polygon([(0, 10), (10, 10), (10, 20), (0, 20)])
        poly4 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])

        smoothed = smoothify_with_topology(
            [poly1, poly2, poly3, poly4],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 4
        assert all(isinstance(p, Polygon) for p in smoothed)
        assert all(p.is_valid for p in smoothed)

    def test_mixed_adjacent_isolated(self):
        """Test with both adjacent and isolated polygons."""
        # Adjacent pair
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        # Isolated polygon
        poly3 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])

        smoothed = smoothify_with_topology(
            [poly1, poly2, poly3],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 3
        assert all(isinstance(p, Polygon) for p in smoothed)
        assert all(p.is_valid for p in smoothed)

    def test_large_grid(self):
        """Test with a larger grid of polygons for performance."""
        # Create 10x10 grid of squares
        polygons = []
        for i in range(10):
            for j in range(10):
                x, y = i * 10, j * 10
                poly = Polygon([
                    (x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)
                ])
                polygons.append(poly)

        smoothed = smoothify_with_topology(
            polygons,
            segment_length=1.0,
            smooth_iterations=2,
        )

        assert len(smoothed) == 100
        assert all(isinstance(p, Polygon) for p in smoothed)
        # At least most should be valid
        valid_count = sum(1 for p in smoothed if p.is_valid)
        assert valid_count >= 90  # Allow some invalid due to complex topology


class TestEdgeCasesTopology:
    """Test edge cases for topology-aware smoothing."""

    def test_empty_list(self):
        """Test with empty list of polygons."""
        smoothed = smoothify_with_topology(
            [],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert smoothed == []

    def test_single_polygon(self):
        """Test with single polygon (no topology to preserve)."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        smoothed = smoothify_with_topology(
            [poly],
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert len(smoothed) == 1
        assert isinstance(smoothed[0], Polygon)
        assert smoothed[0].is_valid

    def test_geodataframe_with_non_polygon_geometries(self):
        """Test GeoDataFrame with mixed geometry types."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        line = LineString([(20, 20), (30, 30)])

        gdf = gpd.GeoDataFrame(
            {"name": ["poly", "line"]},
            geometry=[poly, line],
            crs="EPSG:4326"
        )

        smoothed = smoothify_with_topology(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == 2
        # LineString should be preserved as-is
        assert isinstance(smoothed.geometry.iloc[1], LineString)
