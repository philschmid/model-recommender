from dataclasses import dataclass
from typing import Dict, Optional

from modeler.models.huggingface import HfModel


@dataclass
class SageMakerInference:
    is_llm: bool = False
    min_instance_type: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class TgiInference:
    required_model_memory: Dict[str, int] = None
    additonal_memory_total_tokens: Dict[str, int] = None


@dataclass
class Accelerator:
    type: str
    supports_model_parallelism: bool = False


@dataclass
class InferenceModeler:
    recommended_accelerator: Accelerator
    min_required_memory: Dict[str, int]
    model: HfModel
    sagemaker: SageMakerInference
    tgi: TgiInference


@dataclass
class TrainingModeler:
    pass


@dataclass
class ModelerResponse:
    inference: Optional[InferenceModeler] = None
    training: Optional[TrainingModeler] = None
