FROM python:3.9


WORKDIR /code

# Install dependencies
COPY ./pyproject.toml /code/pyproject.toml
COPY ./recommender /code/recommender
RUN pip install --no-cache-dir --upgrade ".[api]"

# Copy API code
COPY ./api/app /code/app 

# Trust headers from proxy (https://fastapi.tiangolo.com/deployment/docker/#behind-a-tls-termination-proxy)
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]