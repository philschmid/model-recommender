import requests
from huggingface_hub.hf_api import HfApi

from modeler.models.huggingface import HfModel
from modeler.utils.const import SUPPORTED_TGI_MODELS

HF_API = HfApi()


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
        url = f"https://huggingface.co/api/models/{modelId}/revision/{revision}?blobs=1"
        if hub_token:
            headers.update({"Authorization": f"Bearer {hub_token}"})
        req = requests.get(url, headers=headers)

        if req.status_code != 200:
            print(f"Error while getting model info for {modelId}: {req.status_code} , response: {req.text}")
            return None

        res = req.json()

        # filter out models which have custom modelling files
        if any(model["rfilename"].startswith("modeling") for model in res["siblings"]) or any(
            model["rfilename"].endswith("pipeline.py") for model in res["siblings"]
        ):
            print(f"Skipping model {modelId} because it has custom modelling files")
            is_custom_model = True
        else:
            is_custom_model = False

        # model size
        filtered_model_files = [model for model in res["siblings"] if model["rfilename"].startswith("pytorch")]
        model_size = sum([model["size"] for model in filtered_model_files])

        model = HfModel(
            id=res["id"],
            model_type=res["config"]["model_type"],
            task=res["pipeline_tag"],
            library=res["library_name"],
            tags=res["tags"],
            size_in_mb=int(round(model_size / 1024 / 1024, 0)),
            is_custom_model=is_custom_model,
        )

        # widget data
        if "widgetData" in res:
            model.widget_data = res["widgetData"][0]

        # gated
        if "gated" in res:
            model.gated = res["gated"]

        # get license from tags
        if any(tag.startswith("license") for tag in model.tags):
            license_tag = [tag for tag in model.tags if tag.startswith("license")][0]
            model.license = license_tag.split(":")[1]

        return model
    except Exception as e:
        print(f"Error while getting model info for {modelId}: {e}")
        return None


def get_required_memory(model_size: int, accelerator: str = "CPU"):
    """Get the required memory for the model in MB"""
    if accelerator == "CPU":
        return model_size * 2.3
    else:
        return model_size * 1.5


def get_recommended_accelerator(model_size: int):
    """Get the recommended accelerator for the model"""
    if model_size > 1000:
        return "gpu"
    else:
        return "cpu"


def is_model_supported_in_tgi(model_type: str):
    """Check if the model is supported in TGI"""
    return model_type in SUPPORTED_TGI_MODELS
