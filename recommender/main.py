from recommender.utils.calcuation import (
    get_memory_per_model_and_tgi,
    get_model_size,
    get_real_size_with_buffer,
    get_tgi_memory,
)
from recommender.utils.model import get_max_prompt_length, get_quantization_type


def get_tgi_config(
    model_id: str,
    gpu_memory: int,
    num_gpus: int,
):
    """Create a TGI config for a model based on the GPU memory and number of GPUs"""
    quantization_type = get_quantization_type(model_id=model_id)
    max_prompt_length = get_max_prompt_length(model_id=model_id)

    if quantization_type:
        model_memory = get_model_size(model_id=model_id, dtype="int4")
    else:
        model_memory = get_model_size(model_id=model_id, dtype="float16")

    # try different prefill config which cann fit the memory
    for max_prefill_tokens in [2048, 4096, 8192, 10240, 16384, 20480].reverse():
        tgi_additional_memory = get_tgi_memory(
            model_id=model_id, max_prefill_tokens=max_prefill_tokens, dtype="float16"
        )
        real_memory = get_real_size_with_buffer(
            model_memory=model_memory,
            tgi_memory=tgi_additional_memory,
            num_gpus=num_gpus,
        )
        if real_memory < gpu_memory:
            return {
                "model_id": model_id,
                "max_batch_prefill_tokens": max_prefill_tokens,
                "num_gpus": num_gpus,
                "quantization_type": quantization_type,
                "max_prompt_length": max_prompt_length,
            }


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
