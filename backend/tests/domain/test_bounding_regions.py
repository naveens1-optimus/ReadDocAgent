"""Tests for spatial (bounding box / polygon) data on extracted fields."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from domain.entity.extraction_result import BoundingRegion, ExtractedField, Point

#: A page-aligned quadrilateral in Azure's flat form, clockwise from top-left.
QUAD = [1.0, 2.0, 5.0, 2.0, 5.0, 4.0, 1.0, 4.0]


class TestParsingAzurePolygons:
    """Azure sends ``[x1, y1, x2, y2, ...]``; we turn it into points."""

    def test_parses_a_quadrilateral(self) -> None:
        region = BoundingRegion.from_azure({"pageNumber": 2, "polygon": QUAD})

        assert region is not None
        assert region.page_number == 2
        assert [(p.x, p.y) for p in region.polygon] == [
            (1.0, 2.0),
            (5.0, 2.0),
            (5.0, 4.0),
            (1.0, 4.0),
        ]

    def test_accepts_snake_case_keys(self) -> None:
        region = BoundingRegion.from_azure({"page_number": 3, "polygon": QUAD})
        assert region is not None
        assert region.page_number == 3

    def test_accepts_an_sdk_object(self) -> None:
        """The SDK returns attribute-style models, not dicts."""

        class SdkRegion:
            page_number = 1
            polygon = QUAD

        region = BoundingRegion.from_azure(SdkRegion())
        assert region is not None
        assert region.page_number == 1

    def test_parses_more_than_four_points(self) -> None:
        region = BoundingRegion.from_azure(
            {"pageNumber": 1, "polygon": [0, 0, 1, 0, 2, 1, 1, 2, 0, 1]}
        )
        assert region is not None
        assert len(region.polygon) == 5

    def test_drops_an_odd_trailing_coordinate(self) -> None:
        """A truncated response should not discard the whole polygon."""
        region = BoundingRegion.from_azure({"pageNumber": 1, "polygon": [1, 2, 3, 4, 5]})
        assert region is not None
        assert len(region.polygon) == 2

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            {"polygon": QUAD},                  # no page number
            {"pageNumber": 1, "polygon": []},   # no polygon
            {"pageNumber": 1},                  # no polygon at all
            {"pageNumber": 1, "polygon": [1]},  # not even one point
        ],
    )
    def test_unusable_input_returns_none(self, raw: Any) -> None:
        """Not every field carries spatial data; that is normal, not an error."""
        assert BoundingRegion.from_azure(raw) is None

    def test_page_numbers_are_one_based(self) -> None:
        with pytest.raises(ValidationError):
            BoundingRegion(page_number=0, polygon=[Point(x=0, y=0)])


class TestGeometry:
    def test_bounding_box(self) -> None:
        region = BoundingRegion.from_azure({"pageNumber": 1, "polygon": QUAD})
        assert region is not None
        assert region.bounding_box == (1.0, 2.0, 5.0, 4.0)

    def test_bounding_box_of_a_skewed_polygon(self) -> None:
        """A rotated scan gives a polygon that is not axis-aligned."""
        region = BoundingRegion.from_azure(
            {"pageNumber": 1, "polygon": [2, 0, 4, 1, 3, 3, 1, 2]}
        )
        assert region is not None
        assert region.bounding_box == (1.0, 0.0, 4.0, 3.0)


class TestFieldSpatialData:
    def test_field_reports_its_pages(self) -> None:
        regions = [
            BoundingRegion.from_azure({"pageNumber": page, "polygon": QUAD})
            for page in (3, 1, 3)
        ]
        field = ExtractedField(
            name="Terms", bounding_regions=[r for r in regions if r is not None]
        )
        assert field.pages == [1, 3]

    def test_a_value_can_straddle_a_page_break(self) -> None:
        """Azure returns several regions when a value spans pages."""
        regions = [
            BoundingRegion.from_azure({"pageNumber": page, "polygon": QUAD})
            for page in (1, 2)
        ]
        field = ExtractedField(
            name="Terms", bounding_regions=[r for r in regions if r is not None]
        )
        assert field.pages == [1, 2]

    def test_field_without_spatial_data(self) -> None:
        field = ExtractedField(name="VendorName", value="Acme")
        assert field.bounding_regions == []
        assert field.pages == []

    def test_spatial_data_survives_serialisation(self) -> None:
        """The regions are checkpointed and saved, so they must round-trip."""
        region = BoundingRegion.from_azure({"pageNumber": 2, "polygon": QUAD})
        assert region is not None
        field = ExtractedField(
            name="InvoiceTotal", value=495.0, bounding_regions=[region]
        )

        restored = ExtractedField.model_validate(field.model_dump())

        assert restored.bounding_regions[0].bounding_box == (1.0, 2.0, 5.0, 4.0)
        assert restored.pages == [2]
