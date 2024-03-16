import torch
from transformers import AutoConfig
from recommender.utils.const import (
    ACCELERATE_PARSER,
    DEFAULT_BUFFER_PERCENTAGE,
    DEFAULT_PYTORCH_USAGE_PER_GPU,
)
from accelerate.commands.estimate import gather_data
from recommender.utils.utils import get_size_in_gigabytes


def get_tgi_memory(model_id, max_prefill_tokens, dtype):
    """Get the memory required for the TGI model"""
    dtype = "float16" if dtype == "int4" else dtype
    config = AutoConfig.from_pretrained(model_id)
    # calculate the memory required for max prefilled tokens
    tensor_size = (max_prefill_tokens**2) * config.num_attention_heads
    memory = tensor_size * getattr(torch, dtype).itemsize
    return {
        "dtype": dtype,
        "memory_in_bytes": memory,
        "memory_in_gigabytes": get_size_in_gigabytes(memory),
    }


def get_model_size(model_id, dtype):
    args = ACCELERATE_PARSER.parse_args([model_id, "--dtypes", dtype])
    output = gather_data(args)
    model_size = output[0][2] if dtype != "int4" else output[0][2] * 1.5
    return {
        "dtype": dtype,
        "model_size_in_bytes": model_size,
        "model_size_in_gigabytes": get_size_in_gigabytes(model_size),
    }


def get_real_size_with_buffer(model_memory, tgi_memory, num_gpus):
    real_size = (
        model_memory + tgi_memory + (num_gpus * DEFAULT_PYTORCH_USAGE_PER_GPU)
    ) * DEFAULT_BUFFER_PERCENTAGE
    return {
        "real_memory_in_bytes": real_size,
        "real_memory_in_gigabytes": get_size_in_gigabytes(real_size),
    }


def get_memory_per_model_and_tgi(model_id, max_prefill_tokens, dtype, num_gpus=1):
    model_size = get_model_size(model_id, dtype)
    tgi_memory = get_tgi_memory(model_id, max_prefill_tokens, dtype)
    real_memory_with_buffer = get_real_size_with_buffer(
        model_size["model_size_in_bytes"], tgi_memory["memory_in_bytes"], num_gpus
    )

    return {
        "dtype": dtype,
        "real_memory_in_bytes": real_memory_with_buffer["real_memory_in_bytes"],
        "real_memory_in_gigabytes": real_memory_with_buffer["real_memory_in_gigabytes"],
        "real_memory_per_gpu_in_bytes": real_memory_with_buffer["real_memory_in_bytes"]
        / num_gpus,
        "real_memory_per_gpu_in_gigabytes": real_memory_with_buffer[
            "real_memory_in_gigabytes"
        ]
        / num_gpus,
    }
