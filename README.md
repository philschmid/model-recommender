# Hugging Face TGI Recommender

Hugging Face recommender is a utility package to estimate and get helpful information for deploying and training hugging face models.

# API

## Routes 

### GET /v1/tgi/config

**Summary:** Returns Hugging Face Text Generation Configuration

**Query Parameters:**
- `model_id` (string, optional): Model ID
- `gpu_memory` (integer, optional, default=0): GPU memory
- `num_gpus` (integer, optional, default=1): Number of GPUs

### GET /v1/provider/{provider}/recommend

**Summary:** Generates an Instance Type recommendation for a given provider (gcp, huggingface) and returns the instance type and TGI configuration.

**Path Parameters:**
- `provider` (string, required): Provider name (one of: `huggingface`, `gcp`, `aws` (sagemaker))

**Query Parameters:**
- `model_id` (string, optional): Model ID
- `gpu_memory` (integer, optional, default=999): GPU memory

## Development

1. Install the requirements:

```sh
pip install -e ".[api]"
````

2. Run the app with in-memory cache:

```sh
uvicorn app.main:app --reload
```

3. Run the app with redis cache:

```sh
docker-compose up -d
REDIS_URL=redis://localhost:6379 uvicorn app.main:app --reload
```

4. Run the tests:

```
pytest
```

5. Build the app

```sh
docker build -t cache-lookup .
```

6. Run the app:

```sh
docker run -d --name cache-lookup -p 8000:8000 cache-lookup
```

7. curl the app:


To test the dummy route, you can use curl or any HTTP client:

```sh
curl -X GET "http://localhost:8000/dummy?modelid=123"
```

## Test 

You can run the tests with:

```sh
pytest
```

_Note: We use snapshot testing to test. If you change the output of the function, you will need to update the snapshot._

Update all

```
pytest --snapshot-update
```

Update a specific snapshot

```
pytest --snapshot-update tests/test_generate_tgix_snippet.py
```


## Deployment 

Push to main and ArgoCD will deploy the app to the cluster.