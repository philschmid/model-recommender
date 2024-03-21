from accelerate.commands.estimate import estimate_command_parser

ACCELERATE_PARSER = estimate_command_parser()
D_TYPES = ["float32", "float16", "int8", "int4"]

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


SAGEMAKER_INFERENCE_INSTANCE_TYPES = {
    "cpu": [
        {"name": "ml.c6i.large", "memoryInGB": "4"},
        {"name": "ml.c6i.xlarge", "memoryInGB": "8"},
        {"name": "ml.c6i.2xlarge", "memoryInGB": "16"},
        {"name": "ml.c6i.4xlarge", "memoryInGB": "32"},
        {"name": "ml.c6i.8xlarge", "memoryInGB": "64"},
        {"name": "ml.c6i.12xlarge", "memoryInGB": "96"},
        {"name": "ml.c6i.16xlarge", "memoryInGB": "128"},
        {"name": "ml.c6i.24xlarge", "memoryInGB": "192"},
        {"name": "ml.c6i.32xlarge", "memoryInGB": "256"},
        {"name": "ml.m5.large", "memoryInGB": "8"},
        {"name": "ml.m5.xlarge", "memoryInGB": "16"},
        {"name": "ml.m5.2xlarge", "memoryInGB": "32"},
        {"name": "ml.m5.4xlarge", "memoryInGB": "64"},
        {"name": "ml.m5.8xlarge", "memoryInGB": "128"},
        {"name": "ml.m5.12xlarge", "memoryInGB": "192"},
        {"name": "ml.m5.16xlarge", "memoryInGB": "256"},
        {"name": "ml.m5.24xlarge", "memoryInGB": "384"},
        {"name": "ml.c5.large", "memoryInGB": "4"},
        {"name": "ml.c5.xlarge", "memoryInGB": "8"},
        {"name": "ml.c5.2xlarge", "memoryInGB": "16"},
        {"name": "ml.c5.4xlarge", "memoryInGB": "32"},
        {"name": "ml.c5.9xlarge", "memoryInGB": "72"},
        {"name": "ml.c5.12xlarge", "memoryInGB": "96"},
        {"name": "ml.c5.18xlarge", "memoryInGB": "144"},
        {"name": "ml.c5.24xlarge", "memoryInGB": "192"},
    ],
    "gpu": [
        {"name": "ml.g4dn.xlarge", "memoryInGB": "16", "numGpus": "1"},
        {"name": "g4dn.12xlarge", "memoryInGB": "64", "numGpus": "4"},
        {"name": "g4dn.metal", "memoryInGB": "128", "numGpus": "8"},
        {"name": "g5.24xlarge", "memoryInGB": "96", "numGpus": "4"},
        {"name": "g5.48xlarge", "memoryInGB": "192", "numGpus": "8"},
        {"name": "p4d.24xlarge", "memoryInGB": "320", "numGpus": "8"},
        {"name": "p4de.24xlarge", "memoryInGB": "640", "numGpus": "8"},
        # Add more GPU instance types as needed
    ],
}
