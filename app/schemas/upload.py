"""Schemas for the upload endpoint."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after saving uploaded files to disk (data/uploads/)."""

    files_saved: int = Field(..., description="Number of files successfully saved.")
    paths: list[str] = Field(..., description="Relative paths to saved files, e.g. data/uploads/file1.txt")

    model_config = {
        "json_schema_extra": {
            "examples": [{"files_saved": 2, "paths": ["data/uploads/doc1.txt", "data/uploads/sheet.xlsx"]}]
        }
    }
