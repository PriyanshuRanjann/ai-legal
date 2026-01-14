from pydantic import BaseModel, Field, validator

class PDFPageSchema(BaseModel):
    document_id: str = Field(..., description="Stable ID for the source document")
    page_number: int | float = Field(...,description="Page number within the document")
    text: str = Field("", description="Normalized page text")
    source_path: str
    extraction_method: str = Field("pymupdf", description="How the text was obtained")
    char_count: int | float = Field(0, ge=0)

    @validator("char_count", always=True)
    def compute_char_count(cls, value: int, values: dict) -> int:
        if value:
            return value
        return len(values.get("text", ""))

