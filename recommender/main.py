import logging
from recommender.utils.calcuation import (
    get_memory_per_model_and_tgi,
    get_model_size,
    get_real_size_with_buffer,
    get_tgi_memory,
)
from recommender.utils.const import COMMON_GPU_CONFIGS, COMMON_TGI_CONFIGS
from recommender.utils.model import get_max_sequence_length, get_quantization_type

logger = logging.getLogger(__name__)


def get_tgi_config(
    model_id: str,
    gpu_memory: int,
    num_gpus: int = 1,
):
    """Create a TGI config for a model based on the GPU memory and number of GPUs"""
    quantization_type = get_quantization_type(model_id=model_id)
    max_sequence_length = get_max_sequence_length(model_id=model_id)

    if quantization_type:
        model_memory = get_model_size(model_id=model_id, dtype="int4")
    else:
        model_memory = get_model_size(model_id=model_id, dtype="float16")

    # try different prefill config which cann fit the memory
    _configs = COMMON_TGI_CONFIGS.copy()
    _configs.reverse()
    # filter out the configs based on the model max sequence length
    _configs = [c for c in _configs if c["max_total_tokens"] <= max_sequence_length]
    logger.debug(f"Filtered configs: {_configs}")
    for c in _configs:
        print(f"Trying config: {c}")
        tgi_additional_memory = get_tgi_memory(
            model_id=model_id,
            dtype="float16",
            max_input_length=c["max_input_length"],
            max_total_tokens=c["max_total_tokens"],
            max_prefill_tokens=c["max_prefill_tokens"],
        )
        real_memory = get_real_size_with_buffer(
            model_memory=model_memory["model_size_in_bytes"],
            tgi_memory=tgi_additional_memory["memory_in_bytes"],
            num_gpus=num_gpus,
        )
        if real_memory["real_memory_in_gigabytes"] < gpu_memory:
            return {
                "model_id": model_id,
                "max_batch_prefill_tokens": c["max_prefill_tokens"],
                "max_input_length": c["max_input_length"],
                "max_total_tokens": c["max_total_tokens"],
                "num_gpus": num_gpus,
                "quantization_type": quantization_type,
                "estimated_memory_in_gigabytes": real_memory[
                    "real_memory_in_gigabytes"
                ],
            }
    print(f"Could not find a TGI config for {model_id}")
    return None


def get_recommendation(model_id: str):
    """Get the recommendation for a model"""
    recommendations = []
    for gpu_config in COMMON_GPU_CONFIGS:
        tgi_config = get_tgi_config(
            model_id=model_id,
            gpu_memory=gpu_config["gpu_memory"],
            num_gpus=gpu_config["num_gpus"],
        )
        if tgi_config:
            recommendations.append(tgi_config)
    return recommendations


def validate_tgi_config(
    model_id: str,
    max_batch_prefill_tokens: int,
    gpu_memory: int,
    num_gpus: int,
    dtype: str = "float16",
):
    """Validates if a TGI config fits on my GPU"""
    needed_memory = get_memory_per_model_and_tgi(
        model_id=model_id,
        max_prefill_tokens=max_batch_prefill_tokens,
        dtype=dtype,
        num_gpus=num_gpus,
    )
    if needed_memory["real_memory_per_gpu_in_bytes"] > gpu_memory:
        return False
    return True


def get_aws_instance_type(model_id: str, revision: str = "main", hub_token: str = None):
    """Validates if a TGI config fits on my GPU"""
    pass
