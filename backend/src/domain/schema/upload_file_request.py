"""Request schema for an uploaded document."""

from __future__ import annotations

from pathlib import PurePath

from pydantic import BaseModel, Field, field_validator

__all__ = ["SUPPORTED_EXTENSIONS", "UploadFileRequest"]

#: File types Azure Document Intelligence can analyse.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heif"}
)


class UploadFileRequest(BaseModel):
    """Metadata for one uploaded file, validated before any Azure call."""

    file_name: str = Field(min_length=1, max_length=255)
    content_type: str | None = None
    size_bytes: int = Field(ge=1, description="Payload size; must not be empty.")

    @field_validator("file_name")
    @classmethod
    def _check_file_name(cls, value: str) -> str:
        """Reduce to a basename and require a supported extension.

        The directory part is stripped because the name is used to build a
        blob path.
        """
        name = PurePath(value.strip().replace("\\", "/")).name
        if not name or name in {".", ".."}:
            raise ValueError(f"invalid file name: {value!r}")

        extension = PurePath(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"unsupported file type {extension or '(none)'!r}; "
                f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        return name

    @property
    def extension(self) -> str:
        """Lowercased extension, including the dot."""
        return PurePath(self.file_name).suffix.lower()

    @property
    def is_pdf(self) -> bool:
        """PDFs need rendering to an image before the vision model sees them."""
        return self.extension == ".pdf"

    def ensure_within_size_limit(self, max_bytes: int) -> None:
        """Raise if the file is larger than the configured limit."""
        if self.size_bytes > max_bytes:
            raise ValueError(
                f"file is {self.size_bytes / 1_048_576:.1f} MB, over the "
                f"{max_bytes / 1_048_576:.0f} MB limit"
            )
