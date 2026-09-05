"""Fields extracted from a document, each with its own confidence score."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["Point", "BoundingRegion", "ExtractedField", "ExtractionResult"]


class Point(BaseModel):
    """A point in page coordinates."""

    x: float
    y: float


class BoundingRegion(BaseModel):
    """Where on the page a value was found.

    Azure reports the polygon as a flat array -- ``[x1, y1, x2, y2, ...]``,
    normally four points clockwise from the top-left -- and a *list* of these
    per field, because a value can straddle a page break. Page number and
    polygon therefore have to travel together.

    Coordinates are in the page's own unit (inches for PDFs, pixels for
    images).
    """

    page_number: int = Field(ge=1, description="1-based page number.")
    polygon: list[Point] = Field(min_length=1)

    @classmethod
    def from_azure(cls, region: Any) -> BoundingRegion | None:
        """Build from an SDK region or the equivalent dict.

        Returns ``None`` when page number or polygon is missing. Not every
        field carries spatial data, so that is normal rather than an error.
        """
        if region is None:
            return None

        if isinstance(region, dict):
            page = region.get("pageNumber", region.get("page_number"))
            flat = region.get("polygon")
        else:
            page = getattr(region, "page_number", None)
            flat = getattr(region, "polygon", None)

        if page is None or not flat:
            return None

        coordinates = list(flat)
        # An odd trailing coordinate means a truncated response; drop it
        # rather than discarding the whole polygon.
        if len(coordinates) % 2:
            coordinates = coordinates[:-1]
        if len(coordinates) < 2:
            return None

        points = [
            Point(x=float(coordinates[i]), y=float(coordinates[i + 1]))
            for i in range(0, len(coordinates), 2)
        ]
        return cls(page_number=int(page), polygon=points)

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds as ``(min_x, min_y, max_x, max_y)``.

        Useful for highlighting the field, and for a skewed scan where the
        polygon itself is not axis-aligned.
        """
        xs = [point.x for point in self.polygon]
        ys = [point.y for point in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))


class ExtractedField(BaseModel):
    """One field Document Intelligence found, with its own confidence."""

    name: str
    value: Any = None

    #: Document Intelligence's confidence in this field alone. ``None`` when
    #: the model does not report one for the field -- not every field type
    #: carries a score, so absent is different from low.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    #: The raw text as it appeared on the page, before Azure normalised it.
    content: str | None = None

    #: Where on the page the value was found. Empty when Document
    #: Intelligence reports no spatial data for this field.
    bounding_regions: list[BoundingRegion] = Field(default_factory=list)

    #: True once a reviewer changed the value at the approval step.
    edited: bool = False

    @property
    def pages(self) -> list[int]:
        """Pages this field appears on, ascending."""
        return sorted({region.page_number for region in self.bounding_regions})


class ExtractionResult(BaseModel):
    """Everything the extraction step produced for one document."""

    model_id: str = Field(description="Prebuilt model used, e.g. prebuilt-invoice.")
    fields: list[ExtractedField] = Field(default_factory=list)
    page_count: int = 0

    def to_json(self) -> dict[str, Any]:
        """The plain ``{name: value}`` JSON -- what a reviewer edits and what
        is ultimately saved."""
        return {field.name: field.value for field in self.fields}

    def apply_edits(self, edited: dict[str, Any]) -> ExtractionResult:
        """Return a copy with the reviewer's JSON applied.

        Fields the reviewer changed are marked ``edited`` so the saved output
        distinguishes an extracted value from a corrected one. Fields absent
        from ``edited`` were deleted by the reviewer; new keys are added with
        no confidence, since nothing extracted them.
        """
        kept = [
            field.model_copy(
                update={"value": edited[field.name], "edited": True}
            )
            if field.name in edited and edited[field.name] != field.value
            else field.model_copy()
            for field in self.fields
            if field.name in edited
        ]
        known = {field.name for field in self.fields}
        added = [
            ExtractedField(name=name, value=value, edited=True)
            for name, value in edited.items()
            if name not in known
        ]
        return self.model_copy(update={"fields": kept + added})

    @property
    def edited_field_names(self) -> list[str]:
        """Names of fields a reviewer changed or added."""
        return [field.name for field in self.fields if field.edited]

    def low_confidence(self, threshold: float) -> list[str]:
        """Names of scored fields below ``threshold``."""
        return [
            field.name
            for field in self.fields
            if field.confidence is not None and field.confidence < threshold
        ]
