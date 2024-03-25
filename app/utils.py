from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class HfBaseModel(BaseModel):
    class Config:
        protected_namespaces = ()


def handle_exception(e: Exception):
    error_messages = str(e)
    if "`trust_remote_code=True`" in str(e):
        error_messages = "Model requires custom code. Due to security reasons, we cannot provide a recommendation."
    elif "gated repo" in str(e):
        error_messages = "Model is not public, cannot generate recommendation"

    return JSONResponse({"error": error_messages}, status_code=400)
