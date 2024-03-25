FROM python:3.9


WORKDIR /tmp

# Copy all required files and install dependencies
COPY ./pyproject.toml README.md /tmp/
COPY recommender/ /tmp/recommender/
RUN pip install --no-cache-dir --upgrade ".[api]" && \
  rm -rf /tmp/*

# Copy API code
WORKDIR /code
COPY ./api/app /code/app 

# Trust headers from proxy (https://fastapi.tiangolo.com/deployment/docker/#behind-a-tls-termination-proxy)
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]