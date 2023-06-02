from modeler.models.response import InferenceModeler, ModelerResponse
from modeler.utils.huggingface_utils import (
    get_model_info,
    get_recommended_accelerator,
    get_required_memory,
    is_model_supported_in_tgi,
)
from modeler.utils.sagemaker_utils import get_sagemaker_info


def modeler(model_id: str, revision: str = "main", hub_token: str = None) -> ModelerResponse:
    # get model info
    hf_model = get_model_info(model_id, revision, hub_token)
    # check if model is supported in TGI
    is_tgi_supported = is_model_supported_in_tgi(hf_model.model_type)
    if is_tgi_supported:
        rec_accelerator = "gpu"
    else:
        # get recommended accelerator
        rec_accelerator = get_recommended_accelerator(hf_model.size_in_mb)
    # get min required memory
    min_required_memory = get_required_memory(hf_model.size_in_mb, rec_accelerator)

    # sagemaker
    sagemaker = get_sagemaker_info(hf_model, rec_accelerator)

    # construct response object
    res = ModelerResponse(
        inference=InferenceModeler(
            min_required_memory=min_required_memory,
            is_custom_model=hf_model.is_custom_model,
            is_gated=hf_model.gated,
            is_tgi_supported=is_tgi_supported,
            recommended_accelerator=rec_accelerator,
            sagemaker=sagemaker,
        )
    )
    return res
