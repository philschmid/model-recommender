import logging
from dataclasses import dataclass
from typing import Optional

import torch
from accelerate.commands.estimate import gather_data
from transformers import AutoConfig

from recommender.utils.const import (
    ACCELERATE_PARSER,
    DEFAULT_BUFFER_PERCENTAGE,
    DEFAULT_PYTORCH_USAGE_PER_GPU,
    TRUST_REMOTE_CODE,
)
from recommender.utils.utils import get_size_in_gigabytes

logger = logging.getLogger(__name__)


@dataclass
class TGIConfig:
    model_id: str
    max_batch_prefill_tokens: int
    max_input_length: int
    max_total_tokens: int
    num_shard: int
    quantize: str
    estimated_memory_in_gigabytes: float


@dataclass
class MemoryObject:
    in_bytes: int
    dtype: Optional[str] = None

    @property
    def in_gb(self):
        return get_size_in_gigabytes(self.in_bytes)


def get_tgi_memory(
    model_id: str = None,
    max_prefill_tokens: int = None,
    max_total_tokens: int = None,
    max_input_length: int = None,
    dtype: str = None,
):
    """Get the memory required for the TGI model based on the model_id, max_prefill_tokens and max_total_tokens"""
    dtype = "float16" if dtype == "int4" else dtype
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=TRUST_REMOTE_CODE)

    # calculate the memory required for max prefilled tokens
    prefill_tensor_size = (max_prefill_tokens**2) * config.num_attention_heads
    prefill_memory = prefill_tensor_size * getattr(torch, dtype).itemsize
    logger.debug(f"Required memory for prefill: {prefill_memory}")
    # calculate the memory required for max total tokens
    max_tensor_size = (max_total_tokens - max_input_length) * config.hidden_size * config.num_attention_heads
    num_requests = 4  # used in TGI for warmup
    max_memory = max_tensor_size * num_requests * getattr(torch, dtype).itemsize
    logger.debug(f"Required memory for max total tokens: {max_memory}")
    # summarize the memory required for the TGI model
    memory = prefill_memory + max_memory  #
    if config.model_type == "gemma":
        memory = memory * 2

    return MemoryObject(dtype=dtype, in_bytes=memory)


def get_model_size(model_id: str, dtype: str):
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=TRUST_REMOTE_CODE)
    # check max_position_embeddings to avoid out of memory
    if config.max_position_embeddings > 100_000:
        raise ValueError(
            f"max_position_embeddings is too high: {config.max_position_embeddings}, could lead to out of memory"
        )

    args = ACCELERATE_PARSER.parse_args([model_id, "--dtypes", dtype])
    output = gather_data(args)
    model_size = output[0][2] if dtype != "int4" else output[0][2] * 1.5
    return MemoryObject(dtype=dtype, in_bytes=model_size)


def get_real_size_with_buffer(model_memory: int, tgi_memory: int, num_gpus: int):
    real_size = (model_memory + tgi_memory + (num_gpus * DEFAULT_PYTORCH_USAGE_PER_GPU)) * DEFAULT_BUFFER_PERCENTAGE
    return MemoryObject(dtype="float16", in_bytes=real_size)


def get_memory_per_model_and_tgi(model_id: str, max_prefill_tokens: int, dtype: str, num_gpus=1):
    model_size = get_model_size(model_id, dtype)
    tgi_memory = get_tgi_memory(model_id, max_prefill_tokens, dtype)
    real_memory_with_buffer = get_real_size_with_buffer(model_size.in_bytes, tgi_memory.in_bytes, num_gpus)

    return MemoryObject(dtype=dtype, in_bytes=real_memory_with_buffer.in_bytes)
