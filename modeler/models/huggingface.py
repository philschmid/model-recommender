from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HfModel:
    id: str
    model_type: str
    task: str
    library: str
    tags: List[str]
    gated: bool = False
    is_custom_model: bool = False
    size_in_mb: Optional[int] = 999999
    widget_data: Optional[dict] = None
    license: Optional[str] = None
