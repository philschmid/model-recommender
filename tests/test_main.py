from recommender.main import get_tgi_config


def test_config_llama():
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    assert get_tgi_config(model_id, gpu_memory=24_000, num_gpus=1) is not None
