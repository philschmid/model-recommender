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
    "mixtral": 4096,
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
