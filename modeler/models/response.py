from dataclasses import dataclass
from typing import Optional


@dataclass
class SageMakerInference:
    min_instance_type: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class TgiInference:
    is_supported: bool = False
    min_required_memory_in_mb: Optional[str] = None


@dataclass
class InferenceModeler:
    recommended_accelerator: str
    min_required_memory: int
    sagemaker: SageMakerInference
    tgi: TgiInference
    is_custom_model: Optional[bool] = False
    is_gated: Optional[bool] = False


@dataclass
class TrainingModeler:
    pass


@dataclass
class ModelerResponse:
    inference: Optional[InferenceModeler] = None
    training: Optional[TrainingModeler] = None
