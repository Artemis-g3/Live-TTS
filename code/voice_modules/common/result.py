from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackendResult:
    success: bool
    message: str = ""
    output_paths: dict[str, str] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "output_paths": self.output_paths,
            "logs": self.logs,
            "data": self.data,
        }

