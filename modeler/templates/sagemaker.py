import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

# Hub Model configuration. https://huggingface.co/models
hub = {
  "HF_MODEL_ID": "{{ model_id }}",
  "HF_TASK": "{{ task }}"{% if needs_remote_code %},
  "HF_TRUST_REMOTE_CODE": "True"{% endif %}
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version="4.26.0",
    pytorch_version="1.13.1",
    py_version="py39",
    env=hub,
    role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
    initial_instance_count=1, instance_type="{instance_type}"  # number of instances  # ec2 instance type
)

predictor.predict(
    {
        {
            "inputs": "The answer to the universe is [MASK].",
        }
    }
)
