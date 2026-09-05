"""The canonical resume entity.

Resumes have no fixed layout and no dedicated prebuilt model, so
``prebuilt-read`` recovers the text and Azure OpenAI fills this schema in.
"""

from __future__ import annotations

from pydantic import Field

from domain.entity.common import EntityModel

__all__ = ["WorkExperience", "Education", "Resume"]


class WorkExperience(EntityModel):
    """One role in the candidate's history."""

    company: str | None = None
    title: str | None = None
    start_date: str | None = Field(
        default=None, description="As written on the document, e.g. 'Jan 2020'."
    )
    end_date: str | None = Field(default=None, description="'Present' if current.")
    summary: str | None = None


class Education(EntityModel):
    """One qualification."""

    institution: str | None = None
    degree: str | None = None
    field_of_study: str | None = None
    year: str | None = None


class Resume(EntityModel):
    """A resume, as the pipeline stores it."""

    # --- Required ---
    full_name: str

    # --- Optional ---
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    summary: str | None = Field(
        default=None, description="The candidate's own summary or objective."
    )
    skills: list[str] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
