import logging
import torch
from transformers import AutoConfig
from recommender.utils.const import (
    ACCELERATE_PARSER,
    DEFAULT_BUFFER_PERCENTAGE,
    DEFAULT_PYTORCH_USAGE_PER_GPU,
)
from accelerate.commands.estimate import gather_data
from recommender.utils.utils import get_size_in_gigabytes

logger = logging.getLogger(__name__)


def get_tgi_memory(
    model_id: str = None,
    max_prefill_tokens: int = None,
    max_total_tokens: int = None,
    max_input_length: int = None,
    dtype: str = None,
):
    """Get the memory required for the TGI model based on the model_id, max_prefill_tokens and max_total_tokens"""
    dtype = "float16" if dtype == "int4" else dtype
    config = AutoConfig.from_pretrained(model_id)
    # calculate the memory required for max prefilled tokens
    prefill_tensor_size = (max_prefill_tokens**2) * config.num_attention_heads
    prefill_memory = prefill_tensor_size * getattr(torch, dtype).itemsize
    logger.debug(f"Required memory for prefill: {prefill_memory}")
    # calculate the memory required for max total tokens
    max_tensor_size = (
        (max_total_tokens - max_input_length)
        * config.hidden_size
        * config.num_attention_heads
    )
    num_requests = 4  # used in TGI for warmup
    max_memory = max_tensor_size * num_requests * getattr(torch, dtype).itemsize
    logger.debug(f"Required memory for max total tokens: {max_memory}")
    # summarize the memory required for the TGI model
    memory = prefill_memory + max_memory
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
