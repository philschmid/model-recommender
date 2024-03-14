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
    num_gpus: int = 1,
):
    """Create a TGI config for a model based on the GPU memory and number of GPUs"""
    quantization_type = get_quantization_type(model_id=model_id)
    max_prompt_length = get_max_prompt_length(model_id=model_id)

    if quantization_type:
        model_memory = get_model_size(model_id=model_id, dtype="int4")
    else:
        model_memory = get_model_size(model_id=model_id, dtype="float16")

    # try different prefill config which cann fit the memory
    prefill_configs = [4096, 8192, 16384, 32768]
    prefill_configs.reverse()
    for max_prefill_tokens in prefill_configs:
        tgi_additional_memory = get_tgi_memory(
            model_id=model_id, max_prefill_tokens=max_prefill_tokens, dtype="float16"
        )
        real_memory = get_real_size_with_buffer(
            model_memory=model_memory["model_size_in_bytes"],
            tgi_memory=tgi_additional_memory["memory_in_bytes"],
            num_gpus=num_gpus,
        )
        if real_memory["real_memory_in_gigabytes"] < gpu_memory:
            return {
                "model_id": model_id,
                "max_batch_prefill_tokens": max_prefill_tokens,
                "num_gpus": num_gpus,
                "quantization_type": quantization_type,
                "max_prompt_length": max_prompt_length,
                "estimated_memory_in_gigabytes": real_memory[
                    "real_memory_in_gigabytes"
                ],
            }
    print(f"Could not find a TGI config for {model_id}")
    return None


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


def get_recommendation(model_id: str):
    """Get the recommendation for a model"""
    quantization_type = get_quantization_type(model_id=model_id)
    max_prompt_length = get_max_prompt_length(model_id=model_id)

    if quantization_type:
        model_memory = get_model_size(model_id=model_id, dtype="int4")
    else:
        model_memory = get_model_size(model_id=model_id, dtype="float16")

    # try different prefill config which cann fit the memory
    prefill_configs = [4096, 8192, 16384, 32768]
    num_gpu_memory_configs = [24, 40, 48, 80, 96, 160, 192, 320, 384, 640]
    recommendations = []
    for max_prefill_tokens in prefill_configs:
        for potential_memory in num_gpu_memory_configs:
            tgi_additional_memory = get_tgi_memory(
                model_id=model_id,
                max_prefill_tokens=max_prefill_tokens,
                dtype="float16",
            )
            real_memory = get_real_size_with_buffer(
                model_memory=model_memory["model_size_in_bytes"],
                tgi_memory=tgi_additional_memory["memory_in_bytes"],
                num_gpus=1,
            )
            if real_memory["real_memory_in_gigabytes"] < potential_memory:
                break
        recommendations.append(
            {
                "model_id": model_id,
                "max_batch_prefill_tokens": max_prefill_tokens,
                "quantization_type": quantization_type,
                "max_prompt_length": max_prompt_length,
                "required_gpu_memory": real_memory["real_memory_in_gigabytes"],
            }
        )
    return recommendations


def get_aws_instance_type(model_id: str, revision: str = "main", hub_token: str = None):
    """Validates if a TGI config fits on my GPU"""
    pass
