import logging
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field

from app.utils import cached_neuron_cache_lookup
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter()

logger = logging.getLogger("uvicorn")


class PathParams(BaseModel):
    modelid: str = Field(None, description="Hugging Face Model ID to lookup")


@router.get("/config/{provider}")
@cache(expire=3600 * 3)  # cache 3 hours
async def lookup(params: PathParams = Depends()):
    if params.modelid == "":
        # Returns 200 response to be cacheable
        return JSONResponse(
            {"error": "No modelid provided", "cached_configs": []}, status_code=200
        )

    try:
        res = await cached_neuron_cache_lookup(params.modelid)
        return JSONResponse({"cached_configs": res}, status_code=200)

    except Exception as e:
        logger.error(f"Error looking up {params.modelid}: {e}")
        if "gated repo" in str(e):
            # Returns 200 response to be cacheable
            return JSONResponse(
                {
                    "error": f"Model {params.modelid} is not public",
                    "cached_configs": [],
                },
                status_code=200,
            )
        # Returns 200 response to be cacheable
        return JSONResponse(
            {"error": f"No cached entries for {params.modelid}", "cached_configs": []},
            status_code=200,
        )
