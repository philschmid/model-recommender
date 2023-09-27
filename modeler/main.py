from modeler.models.response import InferenceModeler, ModelerResponse, TgiInference, Accelerator
from modeler.utils.huggingface_utils import (
    get_model_info,
    get_recommended_accelerator,
    get_required_memory,
)
from modeler.utils.sagemaker_utils import get_sagemaker_info


def modeler(model_id: str, revision: str = "main", hub_token: str = None) -> ModelerResponse:
    # get model info
    hf_model = get_model_info(model_id, revision, hub_token)
    if hf_model.is_tgi_supported:
        rec_accelerator = Accelerator(type="gpu", supports_model_parallelism=True)
        tgi = TgiInference(
            required_model_memory={
                "fp32": hf_model.size_in_bytes_fp32,
                "fp16": hf_model.size_in_bytes_fp32 / 2,
                "int8": hf_model.size_in_bytes_fp32 / 4,
                "int4": hf_model.size_in_bytes_fp32 / 8,
            },
            additonal_memory_total_tokens={
                "1024": None,
                "2048": None,
                "4096": None,
                "8192": None,
                "16384": None,
            },
        )
    else:
        # get recommended accelerator
        tgi = None
        accelerator = get_recommended_accelerator(hf_model.size_in_bytes_fp32)
        rec_accelerator = Accelerator(type=accelerator, supports_model_parallelism=False)

    # sagemaker
    sagemaker = get_sagemaker_info(hf_model, rec_accelerator.type)
    # tgi
    res = ModelerResponse(
        inference=InferenceModeler(
            recommended_accelerator=rec_accelerator,
            min_required_memory=get_required_memory(hf_model.size_in_bytes_fp32),
            model=hf_model,
            tgi=tgi,
            sagemaker=sagemaker,
        )
    )
    return res
