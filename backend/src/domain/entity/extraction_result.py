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

    #: Confidence in this value. Document Intelligence's own score to begin
    #: with -- ``None`` when it reports none, which is different from low --
    #: and 1.0 once a human has approved it.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    #: What Document Intelligence originally scored, kept once approval
    #: raises ``confidence`` to 1.0. Without it the report could no longer
    #: show how well the machine actually read the document.
    original_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    #: The raw text as it appeared on the page, before Azure normalised it.
    content: str | None = None

    #: Where on the page the value was found. Empty when Document
    #: Intelligence reports no spatial data for this field.
    bounding_regions: list[BoundingRegion] = Field(default_factory=list)

    #: True once a reviewer changed the value.
    edited: bool = False

    #: True once a human approved the value, whether or not they changed it.
    verified: bool = False

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

    def merge_edits(self, edits: dict[str, Any]) -> ExtractionResult:
        """Return a copy with ``edits`` applied over the existing fields.

        Two differences from :meth:`apply_edits`:

        * Fields absent from ``edits`` are **kept**. The correction step sends
          only part of the record, so replacing the whole set would silently
          delete everything the reviewer left alone.
        * Every field in ``edits`` is marked verified, **whether or not the
          value changed**. Submitting a value at the correction gate is the
          reviewer saying they have checked it. Requiring a change would trap
          a low-confidence value that happens to be correct: the only way to
          clear it would be to alter it to something wrong and back again.
        """
        merged = [
            field.model_copy(
                update={
                    "value": edits[field.name],
                    "edited": edits[field.name] != field.value,
                    "verified": True,
                    "original_confidence": (
                        field.original_confidence
                        if field.original_confidence is not None
                        else field.confidence
                    ),
                    "confidence": 1.0,
                }
            )
            if field.name in edits
            else field.model_copy()
            for field in self.fields
        ]
        known = {field.name for field in self.fields}
        added = [
            ExtractedField(
                name=name, value=value, edited=True, verified=True, confidence=1.0
            )
            for name, value in edits.items()
            if name not in known
        ]
        return self.model_copy(update={"fields": merged + added})

    def confidences(self) -> dict[str, float | None]:
        """Per-field confidence scores, for the confidence check."""
        return {field.name: field.confidence for field in self.fields}

    def mark_verified(self) -> ExtractionResult:
        """Return a copy with every field approved at full confidence.

        Once a human has looked at the data and approved it, the machine's
        confidence in its own reading is no longer the measure that matters --
        the values carry a person's sign-off. The original scores are kept in
        ``original_confidence`` so the report can still show how well the
        extraction did.
        """
        return self.model_copy(
            update={
                "fields": [
                    field.model_copy(
                        update={
                            "original_confidence": (
                                field.original_confidence
                                if field.original_confidence is not None
                                else field.confidence
                            ),
                            "confidence": 1.0,
                            "verified": True,
                        }
                    )
                    for field in self.fields
                ]
            }
        )

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
