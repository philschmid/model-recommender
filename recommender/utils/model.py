from transformers import AutoConfig, AutoTokenizer

from recommender.utils.const import (
    ARCHICTECTURE_MAX_LENGTH_MAP,
    TGI_SUPPORTED_MODEL_TYPES,
)


def get_quantization_type(model_id: str):
    """Get the quantization type for the model"""
    config = AutoConfig.from_pretrained(model_id)
    if getattr(config, "quantization_config", None):
        return config.quantization_config["quant_method"]
    elif "gptq" in model_id.lower():
        return "gptq"
    elif "awq" in model_id.lower():
        return "awq"
    else:
        return None


def get_max_sequence_length(model_id: str):
    """Get the max prompt length for the model"""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.model_max_length < 100_000:
        return tokenizer.model_max_length
    else:
        config = AutoConfig.from_pretrained(model_id)
        return ARCHICTECTURE_MAX_LENGTH_MAP.get(config.model_type, 2048)


def is_tgi_supported(model_id: str):
    """Check if the model is supported by TGI"""
    config = AutoConfig.from_pretrained(model_id)
    return config.model_type in TGI_SUPPORTED_MODEL_TYPES
