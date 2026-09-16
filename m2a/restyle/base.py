"""Stage 3: Abstract Base Class and data contracts for generative restyle endpoints."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class EndpointErrorType(str, Enum):
    """Categorization of remote API failure modes."""

    RETRYABLE = "retryable"
    FATAL = "fatal"
    CONTENT_POLICY = "content_policy"
    BUDGET_EXCEEDED = "budget_exceeded"


class EndpointException(Exception):
    """Normalized exception thrown by generative endpoint wrappers."""

    def __init__(self, message: str, error_type: EndpointErrorType, status_code: Optional[int] = None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class CandidateResult(BaseModel):
    """Artifact contract returned by a generative restyle execution."""

    audio_path: str
    endpoint_name: str
    prompt: str
    negative_prompt: str
    strength: float = Field(ge=0.0, le=1.0)
    seed: int
    duration_sec: float
    cost_usd: float = Field(default=0.0, ge=0.0)
    raw_response_metadata: Dict[str, Any] = Field(default_factory=dict)


class RestyleEndpoint(ABC):
    """Abstract Base Class for all generative audio-to-audio endpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def transform(
        self,
        audio_path: Path | str,
        prompt: str,
        negative_prompt: str = "",
        strength: float = 0.75,
        seed: int = 42,
        extra: Optional[Dict[str, Any]] = None,
    ) -> CandidateResult:
        """Execute audio-to-audio transformation on the input stem."""
        pass