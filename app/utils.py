from fastapi_cache.decorator import cache

from optimum.neuron.utils import get_hub_cached_entries


@cache(expire=3600 * 3)  # cache 3 hours
def cached_neuron_cache_lookup(model_id: str):
    return get_hub_cached_entries(model_id)
