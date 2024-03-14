from transformers import AutoConfig

from recommender.utils.const import ARCHICTECTURE_MAX_LENGTH_MAP


def get_quantization_type(model_id: str):
    """Get the quantization type for the model"""
    if "gptq" in model_id.lower():
        return "gptq"
    elif "awq" in model_id.lower():
        return "awq"
    else:
        return None


def get_max_prompt_length(model_id: str):
    """Get the max prompt length for the model"""
    config = AutoConfig.from_pretrained(model_id)
    return ARCHICTECTURE_MAX_LENGTH_MAP.get(config.model_type, 2048)
