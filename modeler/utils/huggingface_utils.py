from typing import Union
import requests
from accelerate.commands.estimate import estimate_command_parser, gather_data
from huggingface_hub.hf_api import HfApi

from modeler.models.huggingface import HfModel

HF_API = HfApi()

parser = estimate_command_parser()

# def validate_model_task(model: ModelInfo):
#     """Validate if the model task is supported by the modeler"""
#     library = model.__dict__.get("library_name", None)

#     if model.pipeline_tag in SUPPORTED_TASKS and library == "transformers":
#         return True
#     return False


def get_model_info(modelId, revision: str = "main", hub_token: str = None):
    """Get model info from huggingface.co API"""
    headers = requests.utils.default_headers()
    headers.update({"User-Agent": "is_ci"})
    try:
        url = f"https://huggingface.co/api/models/{modelId}/revision/{revision}"
        if hub_token:
            headers.update({"Authorization": f"Bearer {hub_token}"})
        req = requests.get(url, headers=headers)

        if req.status_code != 200:
            print(f"Error while getting model info for {modelId}: {req.status_code} , response: {req.text}")
            return None
        res = req.json()

        # filter out models which have custom modelling files
        is_custom_model = True if "custom_code" in res["tags"] else False
        # TGI support
        is_tgi_supported = True if "text-generation-inference" in res["tags"] else False
        is_gated = True if "gated" in res["tags"] else False

        # raw model size
        model_size_in_bytes_fp32 = get_model_size(modelId, "float32")["model_size_in_bytes"]
        # model_size_in_bytes_fp16 = model_size_in_bytes_fp32 / 2
        # model_size_in_bytes_int8 = model_size_in_bytes_fp32 / 4
        # model_size_in_bytes_int4 = model_size_in_bytes_fp32 / 8

        # model data class
        model = HfModel(
            id=res["id"],
            model_type=res["config"]["model_type"],
            task=res["pipeline_tag"],
            library=res["library_name"],
            tags=res["tags"],
            size_in_bytes_fp32=model_size_in_bytes_fp32,
            is_custom_model=is_custom_model,
            is_tgi_supported=is_tgi_supported,
            is_gated=is_gated,
        )

        # widget data
        if "widgetData" in res:
            model.widget_data = res["widgetData"][0]

        # get license from tags
        if any(tag.startswith("license") for tag in model.tags):
            license_tag = [tag for tag in model.tags if tag.startswith("license")][0]
            model.license = license_tag.split(":")[1]

        return model
    except Exception as e:
        print(f"Error while getting model info for {modelId}: {e}")
        return None


def get_model_size(model_id: str, dtype: str):
    args = parser.parse_args([model_id, "--dtypes", dtype])
    output = gather_data(args)
    return {
        "dtype": dtype,
        "model_size_in_bytes": output[0][2],
        "model_size_in_megabytes": output[0][2] / (1024 * 1024),
        "infernece_size_in_bytes": output[0][2] * 1.2,
        "infernece_size_in_megabytes": output[0][2] * 1.2 / (1024 * 1024),
    }


def get_required_memory(model_size: int):
    """Get the required memory for the model in MB"""
    return model_size * 1.2


def get_recommended_accelerator(model_size: int):
    """Get the recommended accelerator for the model"""
    # > 1GB -> GPU
    if model_size > 1_000_000_000:
        return "gpu"
    else:
        return "cpu"
