import logging
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

router = APIRouter()

logger = logging.getLogger("uvicorn")


class PathParams(BaseModel):
    modelid: str = Field(None, description="Hugging Face Model ID to lookup")


@router.get("/recommendation/{provider}/model/{modelid}")
@cache(expire=3600 * 3)  # cache 3 hours
async def lookup(params: PathParams = Depends()):
    if params.model_id == "":
        # Returns 200 response to be cacheable
        return JSONResponse(
            {"error": "No modelid provided", "cached_configs": []}, status_code=200
        )
