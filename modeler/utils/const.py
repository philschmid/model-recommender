SUPPORTED_TGI_MODELS = [
    "bloom",
    "t5",
    "mt5",
    "gpt_neox",
    "gpt2",
    "llama",
    "gpt_bigcode",
    "RefinedWeb",
    "RefinedWebModel",
]

SAGEMAKER_INFERENCE_INSTANCE_TYPES = {
    "cpu": [
        {"name": "c6i.large", "memoryInGB": "4"},
        {"name": "c6i.xlarge", "memoryInGB": "8"},
        {"name": "c6i.2xlarge", "memoryInGB": "16"},
        {"name": "c6i.4xlarge", "memoryInGB": "32"},
        {"name": "c6i.8xlarge", "memoryInGB": "64"},
        {"name": "c6i.12xlarge", "memoryInGB": "96"},
        {"name": "c6i.16xlarge", "memoryInGB": "128"},
        {"name": "c6i.24xlarge", "memoryInGB": "192"},
        {"name": "c6i.32xlarge", "memoryInGB": "256"},
        {"name": "m5.large", "memoryInGB": "8"},
        {"name": "m5.xlarge", "memoryInGB": "16"},
        {"name": "m5.2xlarge", "memoryInGB": "32"},
        {"name": "m5.4xlarge", "memoryInGB": "64"},
        {"name": "m5.8xlarge", "memoryInGB": "128"},
        {"name": "m5.12xlarge", "memoryInGB": "192"},
        {"name": "m5.16xlarge", "memoryInGB": "256"},
        {"name": "m5.24xlarge", "memoryInGB": "384"},
        {"name": "c5.large", "memoryInGB": "4"},
        {"name": "c5.xlarge", "memoryInGB": "8"},
        {"name": "c5.2xlarge", "memoryInGB": "16"},
        {"name": "c5.4xlarge", "memoryInGB": "32"},
        {"name": "c5.9xlarge", "memoryInGB": "72"},
        {"name": "c5.12xlarge", "memoryInGB": "96"},
        {"name": "c5.18xlarge", "memoryInGB": "144"},
        {"name": "c5.24xlarge", "memoryInGB": "192"},
    ],
    "gpu": [
        {"name": "ml.g4dn.xlarge", "memoryInGB": "16"},
        {"name": "g4dn.12xlarge", "memoryInGB": "64"},
        {"name": "g4dn.metal", "memoryInGB": "128"},
        {"name": "g5.24xlarge", "memoryInGB": "96"},
        {"name": "g5.48xlarge", "memoryInGB": "192"},
        {"name": "p4d.24xlarge", "memoryInGB": "320"},
        {"name": "p4de.24xlarge", "memoryInGB": "640"},
        # Add more GPU instance types as needed
    ],
}

# | c6i.large    | 2   | 4   | Nur EBS | Bis zu 12,5 | Bis zu 10 |
# | c6i.xlarge   | 4   | 8   | Nur EBS | Bis zu 12,5 | Bis zu 10 |
# | c6i.2xlarge  | 8   | 16  | Nur EBS | Bis zu 12,5 | Bis zu 10 |
# | c6i.4xlarge  | 16  | 32  | Nur EBS | Bis zu 12,5 | Bis zu 10 |
# | c6i.8xlarge  | 32  | 64  | Nur EBS | 12,5        | 10        |
# | c6i.12xlarge | 48  | 96  | Nur EBS | 18,75       | 15        |
# | c6i.16xlarge | 64  | 128 | Nur EBS | 25          | 20        |
# | c6i.24xlarge | 96  | 192 | Nur EBS | 37,5        | 30        |
# | c6i.32xlarge | 128 | 256 | Nur EBS | 50          | 40        |
# | m5.large    | 2  | 8   | Nur EBS | Bis zu 10 | Bis zu 4.750 |
# | m5.xlarge   | 4  | 16  | Nur EBS | Bis zu 10 | bis zu 4.750 |
# | m5.2xlarge  | 8  | 32  | Nur EBS | Bis zu 10 | bis zu 4.750 |
# | m5.4xlarge  | 16 | 64  | Nur EBS | Bis zu 10 | 4 750        |
# | m5.8xlarge  | 32 | 128 | Nur EBS | 10        | 6.800        |
# | m5.12xlarge | 48 | 192 | Nur EBS | 12        | 9 500        |
# | m5.16xlarge | 64 | 256 | Nur EBS | 20        | 13.600       |
# | m5.24xlarge | 96 | 384 | Nur EBS | 25        | 19.000       |
# | c5.large    | 2  | 4   | Nur EBS | Bis zu 10 | Bis zu 4.750 |
# | c5.xlarge   | 4  | 8   | Nur EBS | Bis zu 10 | Bis zu 4.750 |
# | c5.2xlarge  | 8  | 16  | Nur EBS | Bis zu 10 | Bis zu 4.750 |
# | c5.4xlarge  | 16 | 32  | Nur EBS | Bis zu 10 | 4.750        |
# | c5.9xlarge  | 36 | 72  | Nur EBS | 10        | 9.500        |
# | c5.12xlarge | 48 | 96  | Nur EBS | 12        | 9.500        |
# | c5.18xlarge | 72 | 144 | Nur EBS | 25        | 19.000       |
# | c5.24xlarge | 96 | 192 | Nur EBS | 25        | 19.000       |
# | g4dn.xlarge   | 1 | 4  | 16  | 1 x 125 NVMe-SSD | Bis zu 25 | Bis zu 3,5 | 0,526 USD | 0,316 USD | 0,210 USD |
# | g4dn.12xlarge | 4 | 48 | 192 | 1 x 900 NVMe-SSD | 50        | 9,5        | 3,912 USD | 2,348 USD | 1,564 USD |
# | g4dn.metal    | 8 | 96 | 384 | 2 x 900 NVMe-SSD | 100       | 19         | 7,824 USD | 4,694 USD | 3,130 USD |
# | g5.12xlarge | 4 | 96  | 48  | 192 | 1x3800 | 40  | 16 | $5.672  | $3.403 | $2.269 |
# | g5.24xlarge | 4 | 96  | 96  | 384 | 1x3800 | 50  | 19 | $8.144  | $4.886 | $3.258 |
# | g5.48xlarge | 8 | 192 | 192 | 768 | 2x3800 | 100 | 19 | $16.288 | $9.773 | $6.515 |
# | p4d.24xlarge             | 96 | 1152 | 8 | 320 GB<br>HBM2  | 400 ENA und EFA | Ja | 600 GB/s NVSwitch | 8 x 1000 NVMe-SSD | 19 | 32,77 USD | 19,22 USD | 11,57 USD |
# | p4de.24xlarge (Vorschau) | 96 | 1152 | 8 | 640 GB<br>HBM2e | 400 ENA und EFA | Ja | 600 GB/s NVSwitch | 8 x 1000 NVMe-SSD | 19 | 40,96 USD | 24,01 USD | 14,46 USD |

INFERENCE_SNIPPET_BASIC = """
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {{
	'HF_MODEL_ID':'{model_id}',
	'HF_TASK':'{task}'
}}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.26.0',
	pytorch_version='1.13.1',
	py_version='py39',
	env=hub,
	role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='{instance_type}' # ec2 instance type
)

predictor.predict({{
	"inputs": "The answer to the universe is [MASK].",
}})
"""

INFERENCE_SNIPPET_ZERO_SHOT_CLASSIFICATION = """
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {{
	'HF_MODEL_ID':'{model_id}',
	'HF_TASK':'{task}'
}}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.26.0',
	pytorch_version='1.13.1',
	py_version='py39',
	env=hub,
	role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='{instance_type}' # ec2 instance type
)

predictor.predict({{
    "inputs": "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!",
    "parameters": {"candidate_labels": ["refund", "legal", "faq"]},
}})
"""

INFERENCE_SNIPPET_FILE = """
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import DataSerializer

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {{
	'HF_MODEL_ID':'{model_id}',
	'HF_TASK':'{task}'
}}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	transformers_version='4.26.0',
	pytorch_version='1.13.1',
	py_version='py39',
	env=hub,
	role=role,
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1, # number of instances
	instance_type='{instance_type}' # ec2 instance type
)
predictor.serializer = DataSerializer(content_type='image/x-image') # change to audio/x-audio for audio

with open("cats.jpg", "rb") as f:
	data = f.read()

predictor.predict(data)
"""


SAGEMAKER_INFERENCE_TASK_SNIPPET_MAP = {
    "text-classification": INFERENCE_SNIPPET_BASIC,
    "token-classification": INFERENCE_SNIPPET_BASIC,
    "table-question-answering": INFERENCE_SNIPPET_BASIC,
    "question-answering": INFERENCE_SNIPPET_BASIC,
    "zero-shot-classification": INFERENCE_SNIPPET_ZERO_SHOT_CLASSIFICATION,
    "translation": INFERENCE_SNIPPET_BASIC,
    "summarization": INFERENCE_SNIPPET_BASIC,
    "conversational": INFERENCE_SNIPPET_BASIC,
    "feature-extraction": INFERENCE_SNIPPET_BASIC,
    "text-generation": INFERENCE_SNIPPET_BASIC,
    "text2text-generation": INFERENCE_SNIPPET_BASIC,
    "fill-mask": INFERENCE_SNIPPET_BASIC,
    "sentence-similarity": INFERENCE_SNIPPET_BASIC,
    "automatic-speech-recognition": INFERENCE_SNIPPET_FILE,
    "text-to-speech": INFERENCE_SNIPPET_BASIC,
    "audio-to-audio": INFERENCE_SNIPPET_FILE,
    "audio-classification": INFERENCE_SNIPPET_FILE,
    "image-classification": INFERENCE_SNIPPET_FILE,
    "object-detection": INFERENCE_SNIPPET_FILE,
    "image-segmentation": INFERENCE_SNIPPET_FILE,
}
