from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error", "cancelled"]
    progress: int = 0              # 0–100
    step: Optional[str] = None    # Human-readable current step
    output_url: Optional[str] = None
    error: Optional[str] = None
    # Only authenticated completed AMD results set these. Replay is explicit
    # and must never be presented as inference from a newly submitted image.
    result_origin: Optional[Literal["amd-worker", "amd-replay"]] = None
    result_metadata: Optional[dict[str, Any]] = None


class GenerateFromArtifactRequest(BaseModel):
    """Generic typed-artifact request. Only scene is public in this release."""
    input_kind: Literal["scene"]
    input_path: str
    model_id: str
    collection: str = "Workflows"
    params: dict[str, Any] = Field(default_factory=dict)
