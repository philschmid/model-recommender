def get_recommended_accelerator(model_size: int):
    """Get the recommended accelerator for the model"""
    # > 1GB -> GPU
    if model_size > 1_000_000_000:
        return "gpu"
    else:
        return "cpu"


def get_size_in_gigabytes(size_in_bytes: int):
    """Get the size in GB"""
    return int(size_in_bytes / (1024 * 1024 * 1024))
