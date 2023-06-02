from modeler.models.huggingface import HfModel
from modeler.models.response import SageMakerInference
from modeler.utils.const import SAGEMAKER_INFERENCE_INSTANCE_TYPES, SAGEMAKER_INFERENCE_TASK_SNIPPET_MAP
from modeler.utils.huggingface_utils import get_required_memory


def get_sagemaker_info(model: HfModel, accelerator: str):
    min_memory = get_required_memory(model.size_in_mb, accelerator)
    instance_type = get_min_instance(min_memory, accelerator)
    code_snippet = get_inference_code_snippet(model.id, model.task, instance_type)
    return SageMakerInference(min_instance_type=instance_type, code_snippet=code_snippet)


def get_min_instance(size: int, accelerator: str):
    model_size_in_gb = size / 1024
    # find first instance which is greater than model size for the accelerator
    selected_instance = None
    for instance in SAGEMAKER_INFERENCE_INSTANCE_TYPES[accelerator]:
        if int(instance["memoryInGB"]) > int(model_size_in_gb):
            print(f"Selected instance {instance['name']} for model size {model_size_in_gb} GB")
            selected_instance = instance
            break

    # if no instance found, raise exception
    if not selected_instance:
        raise Exception(
            f"Could not find instance type for model size {model_size_in_gb} GB and accelerator {accelerator}"
        )
    # return instance name
    return selected_instance["name"]


def get_inference_code_snippet(model_id: str, task: str, instance_type: str):
    snippet = SAGEMAKER_INFERENCE_TASK_SNIPPET_MAP[task]
    return snippet.format(model_id=model_id, task=task, instance_type=instance_type)
