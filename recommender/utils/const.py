import os

from accelerate.commands.estimate import estimate_command_parser

ACCELERATE_PARSER = estimate_command_parser()
D_TYPES = ["float32", "float16", "int8", "int4"]

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true"

DEFAULT_PYTORCH_USAGE_PER_GPU = 1_000_000_000
DEFAULT_BUFFER_PERCENTAGE = 1.2

ARCHICTECTURE_MAX_LENGTH_MAP = {
    "llama": 4096,
    "falcon": 2048,
    "gemma": 4096,
    "mistral": 4096,
    "mixtral": 32768,
    "gpt_bigcode": 8192,
    "starcoder2": 16384,
}

TGI_SUPPORTED_MODEL_TYPES = [
    "bloom",
    "t5",
    "mt5",
    "gpt_neox",
    "gpt2",
    "llama",
    "gpt_bigcode",
    "RefinedWeb",
    "RefinedWebModel",
    "falcon",
    "mpt",
    "idefics",
    "opt",
    "mistral",
    "mixtral",
    "gemma",
    "phi",
    "qwen2",
    "starcoder2",
]


COMMON_TGI_CONFIGS = [
    {
        "max_input_length": 800,
        "max_total_tokens": 1024,
        "max_prefill_tokens": 2048,
    },
    {
        "max_input_length": 1024,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 8192,
    },
    {
        "max_input_length": 1024,
        "max_total_tokens": 2048,
        "max_prefill_tokens": 2048,
    },
    {
        "max_input_length": 1512,
        "max_total_tokens": 2048,
        "max_prefill_tokens": 2048,
    },
    {
        "max_input_length": 2048,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 4096,
    },
    {
        "max_input_length": 3072,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 4096,
    },
    {
        "max_input_length": 3072,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 6144,
    },
    {
        "max_input_length": 2048,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 8192,
    },
    {
        "max_input_length": 3072,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 8192,
    },
    {
        "max_input_length": 3072,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 16384,
    },
    {
        "max_input_length": 4000,
        "max_total_tokens": 4096,
        "max_prefill_tokens": 32768,
    },
    {
        "max_input_length": 4096,
        "max_total_tokens": 8192,
        "max_prefill_tokens": 8192,
    },
    {
        "max_input_length": 4096,
        "max_total_tokens": 8192,
        "max_prefill_tokens": 16384,
    },
    {
        "max_input_length": 8000,
        "max_total_tokens": 8192,
        "max_prefill_tokens": 32768,
    },
    {
        "max_input_length": 8000,
        "max_total_tokens": 16384,
        "max_prefill_tokens": 16384,
    },
    {
        "max_input_length": 8000,
        "max_total_tokens": 16384,
        "max_prefill_tokens": 32768,
    },
]

COMMON_GPU_CONFIGS = [
    {
        "num_gpus": 1,
        "gpu_memory": 16,
    },
    {
        "num_gpus": 1,
        "gpu_memory": 24,
    },
    {
        "num_gpus": 1,
        "gpu_memory": 40,
    },
    {
        "num_gpus": 2,
        "gpu_memory": 48,
    },
    {
        "num_gpus": 1,
        "gpu_memory": 80,
    },
    {
        "num_gpus": 4,
        "gpu_memory": 96,
    },
    {
        "num_gpus": 2,
        "gpu_memory": 160,
    },
    {
        "num_gpus": 8,
        "gpu_memory": 192,
    },
    {
        "num_gpus": 4,
        "gpu_memory": 320,
    },
    {
        "num_gpus": 8,
        "gpu_memory": 640,
    },
]

GOOGLE_CLOUD_INFERENCE_INSTANCE_TYPES = {
    "gpu": [
        # order by most optimal based on num gpus and memory
        {"name": "g2-standard-4", "memoryInGB": 24, "numGpus": 1},
        {"name": "g2-standard-24", "memoryInGB": 48, "numGpus": 2},
        {"name": "a2-ultragpu-1g", "memoryInGB": 80, "numGpus": 1},
        {"name": "a2-ultragpu-2g", "memoryInGB": 160, "numGpus": 2},
        {"name": "a2-ultragpu-4g", "memoryInGB": 320, "numGpus": 4},
        {"name": "a2-ultragpu-8g", "memoryInGB": 640, "numGpus": 8},
        {"name": "a3-highgpu-8g", "memoryInGB": 640, "numGpus": 8},
        {"name": "g2-standard-48", "memoryInGB": 96, "numGpus": 4},
        {"name": "g2-standard-96", "memoryInGB": 192, "numGpus": 8},
        {"name": "a2-highgpu-1g", "memoryInGB": 40, "numGpus": 1},
        {"name": "a2-highgpu-2g", "memoryInGB": 80, "numGpus": 2},
        {"name": "a2-highgpu-4g", "memoryInGB": 160, "numGpus": 4},
        {"name": "a2-highgpu-8g", "memoryInGB": 320, "numGpus": 8},
    ]
}


AWS_INFERENCE_INSTANCE_TYPES = {
    "gpu": [
        {"name": "g5.2xlarge", "memoryInGB": 24, "numGpus": 1},
        {"name": "g5.12xlarge", "memoryInGB": 96, "numGpus": 4},
        {"name": "g5.48xlarge", "memoryInGB": 192, "numGpus": 8},
        {"name": "p4d.24xlarge", "memoryInGB": 320, "numGpus": 8},
        {"name": "p4de.24xlarge", "memoryInGB": 640, "numGpus": 8},
        # Add more GPU instance types as needed
    ],
}

HUGGINGFACE_INSTANCE_TYPES = {
    "gpu": [
        {"name": "aws-nvidia-a100-x2", "memoryInGB": 24, "numGpus": 1},
        {"name": "aws-nvidia-a100-x1", "memoryInGB": 80, "numGpus": 1},
        {"name": "aws-nvidia-a100-x2", "memoryInGB": 160, "numGpus": 2},
        {"name": "aws-nvidia-a100-x4", "memoryInGB": 320, "numGpus": 4},
        {"name": "aws-nvidia-a100-x8", "memoryInGB": 640, "numGpus": 8},
        # Add more GPU instance types as needed
    ],
}
