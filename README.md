# Hugging Face TGI Recommender

Hugging Face recommender is a utility package to estimate and get helpful information for deploying and training hugging face models. The package includes: 

* `recommender`: python library to estimate the required resources for a given model, respecting memory, kv-cache and generation. 
* `api`: FastAPI app to expose the recommender as a REST API.
* `gradio`: Gradio app to expose the recommender as a web app.
* `notebooks`: Jupyter notebooks to test and experiment with the recommender, used for our partners, e.g. AWS, Cloudflare, etc.
* `tests`: Unit tests for the recommender, not much here.

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
cd api && uvicorn app.main:app --reload
```

1. Build the app

```sh
docker build -t recommender-api -f api/Dockerfile .
```

6. Run the app:

```sh
docker run -p 8000:8000 recommender-api
```

7. curl the app:


To test the dummy route, you can use curl or any HTTP client:

```sh
curl  -X GET 'localhost:8000/v1/provider/gcp/recommend?model_id=HuggingFaceH4%2Fzephyr-7b-beta'
```


## Deployment 

Push to main and ArgoCD will deploy the app to the cluster.
