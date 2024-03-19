from dataclasses import asdict
import logging
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field

from recommender.main import get_tgi_config
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from recommender.utils.calcuation import TGIConfig

router = APIRouter()

logger = logging.getLogger("uvicorn")


from fastapi_cache.decorator import cache


class PathParams(BaseModel):
    model_id: str = Field(None, description="Hugging Face Model ID to lookup")
    gpu_memory: int = Field(None, description="GPU memory in GB")


@router.get("/tgi/config")
@cache(expire=3600 * 3)  # cache 3 hours
async def config(params: PathParams = Depends()):
    if params.model_id == "" or params.gpu_memory == "":
        return JSONResponse(
            {"error": "No model_id or gpu_memory provided", "config": {}},
            status_code=400,
        )
    gpu_memory = int(params.gpu_memory)
    try:
        tgi = get_tgi_config(params.model_id, gpu_memory)
        if tgi is None:
            return JSONResponse(
                {
                    "error": f"Couldn't generate TGI config for {params.model_id}",
                    "config": {},
                },
                status_code=400,
            )
        return JSONResponse({"config": asdict(tgi)}, status_code=200)

    except Exception as e:
        logger.error(f"Error looking up {params.model_id}: {e}")
        if "gated repo" in str(e):
            # Returns 200 response to be cacheable
            return JSONResponse(
                {
                    "error": f"Model {params.model_id} is not public",
                    "config": {},
                },
                status_code=400,
            )
        # Returns 200 response to be cacheable
        return JSONResponse(
            {"error": str(e), "config": {}},
            status_code=400,
        )
