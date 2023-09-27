from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HfModel:
    id: str
    model_type: str
    task: str
    library: str
    tags: List[str]
    is_custom_model: bool = False
    is_tgi_supported: bool = False
    is_gated: bool = False
    size_in_bytes_fp32: Optional[int] = 999999
    widget_data: Optional[dict] = None
    license: Optional[str] = None
