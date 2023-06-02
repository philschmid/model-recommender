# Hugging Face Modeler

Hugging Face Modeler is a utility package to estimate and get helpful information for deploying and training hugging face models.

## Getting Started

```bash
pip install -e .
```

get info
```bash
modeler -m bert-base-uncased
```

## Features

- [x] recommended inference accelerator
- [x] min required memory
- [x] if the model is supported by Text Generation Inference
- [x] if the model needs remote custom code
- [x] if the model is gated 
- [ ] sagemaker inference
  - [x] min required instance type
  - [ ] inference snippet

