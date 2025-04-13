# Building Production-Ready Applications

Welcome to the eighth installment of our "Mastering Hugging Face Transformers" series! In our previous posts, we've explored transformer models, their architecture, pre-trained models, fine-tuning techniques, specialized NLP tasks, multimodal applications, and efficient transformers. Now, we'll focus on taking these models to production.

In this post, we'll cover model serving strategies, creating APIs with FastAPI, deployment options, monitoring deployed models, and handling data drift and model updates. These topics are crucial for building reliable, scalable applications with transformer models.

## Model Serving Strategies

### Understanding Model Serving

Model serving is the process of making trained models available for inference in production environments. Effective model serving requires balancing several considerations:

- **Performance**: Minimizing latency and maximizing throughput
- **Scalability**: Handling varying loads efficiently
- **Reliability**: Ensuring consistent availability
- **Resource efficiency**: Optimizing compute and memory usage
- **Monitoring**: Tracking performance and detecting issues

### Batch vs. Real-time Inference

There are two main approaches to model serving:

1. **Batch inference**: Processing multiple requests together periodically
2. **Real-time inference**: Processing each request immediately as it arrives

Let's implement both approaches:

```python
# Batch inference example
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import time

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Batch inference function
def batch_inference(texts, batch_size=32):
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process outputs
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().numpy()
        
        # Convert to results
        for j, pred in enumerate(predictions):
            label_id = pred.argmax()
            label = model.config.id2label[label_id]
            score = pred[label_id]
            results.append({"text": batch[j], "label": label, "score": float(score)})
    
    return results

# Real-time inference function
def realtime_inference(text):
    # Tokenize
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process output
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label_id = predictions.argmax().item()
    label = model.config.id2label[label_id]
    score = predictions[label_id].item()
    
    return {"text": text, "label": label, "score": score}

# Test batch inference
test_texts = [
    "I absolutely loved this movie! The acting was superb.",
    "What a waste of time. The plot made no sense at all.",
    "It was okay, not great but not terrible either.",
    # Add more texts...
]

# Measure batch inference time
start_time = time.time()
batch_results = batch_inference(test_texts)
end_time = time.time()
print(f"Batch inference time for {len(test_texts)} texts: {end_time - start_time:.4f} seconds")

# Measure real-time inference time
start_time = time.time()
realtime_results = [realtime_inference(text) for text in test_texts]
end_time = time.time()
print(f"Real-time inference time for {len(test_texts)} texts: {end_time - start_time:.4f} seconds")
```

### Model Caching and Batching

Implementing caching and efficient batching can significantly improve performance:

```python
from functools import lru_cache
import torch

# Cache for tokenization results
@lru_cache(maxsize=1024)
def cached_tokenize(text):
    return tokenizer(text, padding=False, truncation=True, return_tensors="pt")

# Dynamic batching for incoming requests
class DynamicBatcher:
    def __init__(self, model, tokenizer, max_batch_size=32, max_wait_time=0.1):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = []
        self.device = next(model.parameters()).device
    
    def add_request(self, text, request_id):
        self.queue.append((text, request_id))
        
        # Process batch if it's full
        if len(self.queue) >= self.max_batch_size:
            return self.process_batch()
        
        return None
    
    def process_batch(self):
        if not self.queue:
            return {}
        
        # Get current batch
        batch = self.queue
        self.queue = []
        
        # Extract texts and IDs
        texts = [item[0] for item in batch]
        request_ids = [item[1] for item in batch]
        
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().numpy()
        
        # Create results dictionary
        results = {}
        for i, request_id in enumerate(request_ids):
            label_id = predictions[i].argmax()
            label = self.model.config.id2label[label_id]
            score = float(predictions[i][label_id])
            results[request_id] = {"text": texts[i], "label": label, "score": score}
        
        return results

# Example usage
batcher = DynamicBatcher(model, tokenizer)

# Simulate incoming requests
import uuid

for text in test_texts:
    request_id = str(uuid.uuid4())
    results = batcher.add_request(text, request_id)
    
    if results:
        print(f"Processed batch with {len(results)} requests")

# Process any remaining requests
final_results = batcher.process_batch()
if final_results:
    print(f"Processed final batch with {len(final_results)} requests")
```

## Creating APIs with FastAPI

### Setting Up a FastAPI Application

FastAPI is a modern, fast web framework for building APIs with Python. Let's create a simple API for our transformer model:

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import uvicorn
import time
import uuid

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Create FastAPI app
app = FastAPI(title="Transformer API", description="API for sentiment analysis using transformers")

# Define request and response models
class SentimentRequest(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "I really enjoyed this movie. The acting was superb!"
            }
        }

class BatchSentimentRequest(BaseModel):
    texts: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I really enjoyed this movie. The acting was superb!",
                    "This was a terrible waste of time."
                ]
            }
        }

class SentimentResponse(BaseModel):
    text: str
    label: str
    score: float

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    processing_time: float

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str

# Store for async jobs
job_store = {}

# Inference function
def predict_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label_id = predictions.argmax().item()
    label = model.config.id2label[label_id]
    score = predictions[label_id].item()
    
    return {"text": text, "label": label, "score": score}

# Batch inference function
def predict_batch_sentiment(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    results = []
    for i, pred in enumerate(predictions):
        label_id = pred.argmax().item()
        label = model.config.id2label[label_id]
        score = pred[label_id].item()
        results.append({"text": texts[i], "label": label, "score": score})
    
    return results

# Background task for async processing
def process_async_job(job_id, texts):
    try:
        start_time = time.time()
        results = predict_batch_sentiment(texts)
        end_time = time.time()
        
        job_store[job_id] = {
            "status": "completed",
            "results": results,
            "processing_time": end_time - start_time
        }
    except Exception as e:
        job_store[job_id] = {
            "status": "failed",
            "error": str(e)
        }

# API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Transformer API", "model": model_name}

@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        result = predict_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-sentiment", response_model=BatchSentimentResponse)
def analyze_batch_sentiment(request: BatchSentimentRequest):
    try:
        start_time = time.time()
        results = predict_batch_sentiment(request.texts)
        end_time = time.time()
        
        return {
            "results": results,
            "processing_time": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/async-sentiment", response_model=AsyncJobResponse)
def analyze_sentiment_async(request: BatchSentimentRequest, background_tasks: BackgroundTasks):
    try:
        job_id = str(uuid.uuid4())
        job_store[job_id] = {"status": "processing"}
        
        background_tasks.add_task(process_async_job, job_id, request.texts)
        
        return {"job_id": job_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    if job["status"] == "completed":
        return {
            "status": job["status"],
            "results": job["results"],
            "processing_time": job["processing_time"]
        }
    elif job["status"] == "failed":
        return {"status": job["status"], "error": job["error"]}
    else:
        return {"status": job["status"]}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": model_name}

# Run the API server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### API Documentation with Swagger

FastAPI automatically generates interactive API documentation using Swagger UI. When you run the FastAPI application, you can access the documentation at `/docs` (e.g., http://localhost:8000/docs).

### Handling Concurrent Requests

For production applications, you'll need to handle concurrent requests efficiently:

```python
# Add this to your FastAPI application

from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for handling requests
executor = ThreadPoolExecutor(max_workers=4)  # Adjust based on your hardware

# Modified batch endpoint to use the thread pool
@app.post("/concurrent-batch-sentiment", response_model=BatchSentimentResponse)
def analyze_concurrent_batch_sentiment(request: BatchSentimentRequest):
    try:
        start_time = time.time()
        
        # Split the batch into smaller chunks
        batch_size = 8  # Adjust based on your model and hardware
        batches = [request.texts[i:i+batch_size] for i in range(0, len(request.texts), batch_size)]
        
        # Process batches concurrently
        futures = [executor.submit(predict_batch_sentiment, batch) for batch in batches]
        
        # Collect results
        results = []
        for future in futures:
            results.extend(future.result())
        
        end_time = time.time()
        
        return {
            "results": results,
            "processing_time": end_time - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Deployment Options (Cloud, Containers)

### Containerizing with Docker

Containerization makes deployment consistent across different environments. Let's create a Dockerfile for our API:

```dockerfile
# Use an official PyTorch image as base
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

And a requirements.txt file:

```
transformers==4.18.0
torch==1.9.0
fastapi==0.75.0
uvicorn==0.17.6
pydantic==1.9.0
```

Build and run the Docker container:

```bash
# Build the Docker image
docker build -t transformer-api .

# Run the container
docker run -p 8000:8000 transformer-api
```

### Deploying to Cloud Platforms

Let's look at deployment options for different cloud providers:

#### AWS Elastic Container Service (ECS)

```yaml
# task-definition.json
{
  "family": "transformer-api",
  "networkMode": "awsvpc",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "transformer-api",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/transformer-api:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/transformer-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "memory": 4096,
      "cpu": 1024
    }
  ],
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096"
}
```

Deploy to ECS:

```bash
# Push image to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker tag transformer-api:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/transformer-api:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/transformer-api:latest

# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create ECS service
aws ecs create-service --cluster your-cluster --service-name transformer-api --task-definition transformer-api --desired-count 1 --launch-type FARGATE --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678],securityGroups=[sg-12345678],assignPublicIp=ENABLED}"
```

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/your-project/transformer-api

# Deploy to Cloud Run
gcloud run deploy transformer-api --image gcr.io/your-project/transformer-api --platform managed --memory 4Gi --cpu 2 --concurrency 10
```

#### Azure Container Instances

```bash
# Create a resource group
az group create --name transformer-api-rg --location eastus

# Create a container registry
az acr create --resource-group transformer-api-rg --name transformerregistry --sku Basic

# Log in to the registry
az acr login --name transformerregistry

# Tag and push the image
docker tag transformer-api transformerregistry.azurecr.io/transformer-api:latest
docker push transformerregistry.azurecr.io/transformer-api:latest

# Create a container instance
az container create --resource-group transformer-api-rg --name transformer-api --image transformerregistry.azurecr.io/transformer-api:latest --dns-name-label transformer-api --ports 8000
```

### Serverless Deployment

For smaller models or less frequent usage, serverless deployment can be cost-effective:

```python
# AWS Lambda handler (app.py)
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer (done outside the handler for warm starts)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body'])
        text = body.get('text')
        
        if not text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No text provided'})
            }
        
        # Tokenize
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process output
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        label_id = predictions.argmax().item()
        label = model.config.id2label[label_id]
        score = predictions[label_id].item()
        
        # Return result
        return {
            'statusCode': 200,
            'body': json.dumps({
                'text': text,
                'label': label,
                'score': score
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

## Monitoring and Maintaining Deployed Models

### Setting Up Monitoring

Monitoring is essential for ensuring your deployed models are performing as expected:

```python
# Add this to your FastAPI application

from prometheus_client import Counter, Histogram, start_http_server
import time

# Start Prometheus metrics server
start_http_server(8001)  # Metrics available at http://localhost:8001

# Define metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
RESPONSE_TIME = Histogram('api_response_time_seconds', 'Response time in seconds', ['endpoint'])
PREDICTION_DISTRIBUTION = Counter('prediction_distribution', 'Distribution of predictions', ['label'])

# Middleware to track metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record response time
    endpoint = request.url.path
    RESPONSE_TIME.labels(endpoint=endpoint).observe(time.time() - start_time)
    
    # Record request count
    status = response.status_code
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    
    return response

# Modified sentiment endpoint to track prediction distribution
@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        result = predict_sentiment(request.text)
        
        # Record prediction distribution
        PREDICTION_DISTRIBUTION.labels(label=result["label"]).inc()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Logging Predictions and Inputs

Logging inputs and predictions helps with debugging and analysis:

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transformer-api")

# Create a separate logger for predictions
prediction_logger = logging.getLogger("predictions")
prediction_logger.setLevel(logging.INFO)
prediction_handler = logging.FileHandler("predictions.log")
prediction_handler.setFormatter(logging.Formatter('%(message)s'))
prediction_logger.addHandler(prediction_handler)

# Modified sentiment endpoint with logging
@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        # Log the request
        logger.info(f"Received sentiment analysis request: {request.text[:50]}...")
        
        # Make prediction
        start_time = time.time()
        result = predict_sentiment(request.text)
        inference_time = time.time() - start_time
        
        # Log the prediction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": request.text,
            "output": result,
            "inference_time": inference_time
        }
        prediction_logger.info(json.dumps(log_entry))
        
        # Log success
        logger.info(f"Sentiment analysis completed in {inference_time:.4f}s: {result['label']}")
        
        return result
    except Exception as e:
        # Log error
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Model Performance Monitoring

Tracking model performance over time helps detect degradation:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Class for monitoring model performance
class ModelMonitor:
    def __init__(self, model_name, evaluation_interval=timedelta(days=1)):
        self.model_name = model_name
        self.evaluation_interval = evaluation_interval
        self.last_evaluation = datetime.now()
        self.ground_truth_buffer = []
        self.prediction_buffer = []
        self.performance_history = []
    
    def add_prediction(self, prediction, ground_truth=None):
        # If ground truth is available (e.g., from user feedback)
        if ground_truth is not None:
            self.ground_truth_buffer.append(ground_truth)
            self.prediction_buffer.append(prediction)
    
    def evaluate_if_needed(self):
        now = datetime.now()
        if now - self.last_evaluation >= self.evaluation_interval and len(self.ground_truth_buffer) > 0:
            self.evaluate()
            self.last_evaluation = now
    
    def evaluate(self):
        # Calculate metrics
        accuracy = accuracy_score(self.ground_truth_buffer, self.prediction_buffer)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.ground_truth_buffer, self.prediction_buffer, average='weighted'
        )
        
        # Record performance
        performance = {
            "timestamp": datetime.now().isoformat(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sample_size": len(self.ground_truth_buffer)
        }
        self.performance_history.append(performance)
        
        # Log performance
        logger.info(f"Model performance evaluation: {json.dumps(performance)}")
        
        # Check for degradation
        if len(self.performance_history) > 1:
            prev_f1 = self.performance_history[-2]["f1"]
            current_f1 = f1
            if current_f1 < prev_f1 * 0.95:  # 5% degradation
                logger.warning(f"Model performance degradation detected! F1 score dropped from {prev_f1:.4f} to {current_f1:.4f}")
        
        # Reset buffers
        self.ground_truth_buffer = []
        self.prediction_buffer = []
        
        return performance

# Create monitor instance
monitor = ModelMonitor(model_name)

# Add endpoint for user feedback
class FeedbackRequest(BaseModel):
    prediction_id: str
    correct_label: str

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        # Retrieve the original prediction (in a real system, store predictions with IDs)
        # For this example, we'll simulate it
        original_prediction = "POSITIVE"  # This would come from storage in a real system
        
        # Add to monitor
        monitor.add_prediction(original_prediction, request.correct_label)
        
        # Check if evaluation is needed
        monitor.evaluate_if_needed()
        
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## Handling Data Drift and Model Updates

### Detecting Data Drift

Data drift occurs when the distribution of production data differs from training data:

```python
from scipy.stats import ks_2samp
import numpy as np

class DriftDetector:
    def __init__(self, reference_data, feature_names, drift_threshold=0.05):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold
        self.current_window = []
        self.window_size = 1000
    
    def add_observation(self, features):
        self.current_window.append(features)
        
        # Check for drift when window is full
        if len(self.current_window) >= self.window_size:
            return self.check_drift()
        
        return None
    
    def check_drift(self):
        current_data = np.array(self.current_window)
        drift_detected = False
        drift_features = []
        
        # Check each feature for drift
        for i, feature_name in enumerate(self.feature_names):
            # Perform Kolmogorov-Smirnov test
            ks_result = ks_2samp(self.reference_data[:, i], current_data[:, i])
            p_value = ks_result.pvalue
            
            # If p-value is below threshold, drift is detected
            if p_value < self.drift_threshold:
                drift_detected = True
                drift_features.append({
                    "feature": feature_name,
                    "p_value": p_value
                })
        
        # Reset window
        self.current_window = []
        
        if drift_detected:
            logger.warning(f"Data drift detected in features: {[f['feature'] for f in drift_features]}")
        
        return {
            "drift_detected": drift_detected,
            "drift_features": drift_features if drift_detected else None
        }

# In a real application, you would extract features from inputs
# For this example, we'll use embeddings from the model

# Get reference embeddings from training data
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        # Get embeddings from the model's hidden states
        outputs = model.distilbert(inputs["input_ids"], inputs["attention_mask"])
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use [CLS] token
    
    return embeddings

# Initialize with reference data (in practice, use your training data)
reference_texts = [
    "This movie was fantastic!",
    "I hated every minute of it.",
    # Add more reference texts...
]
reference_embeddings = get_embeddings(reference_texts)
feature_names = [f"embedding_{i}" for i in range(reference_embeddings.shape[1])]

drift_detector = DriftDetector(reference_embeddings, feature_names)

# Modified endpoint to check for drift
@app.post("/sentiment", response_model=SentimentResponse)
def analyze_sentiment(request: SentimentRequest):
    try:
        # Make prediction
        result = predict_sentiment(request.text)
        
        # Get embeddings for drift detection
        embedding = get_embeddings([request.text])[0]
        
        # Add to drift detector
        drift_result = drift_detector.add_observation(embedding)
        
        # Log drift if detected
        if drift_result and drift_result["drift_detected"]:
            logger.warning("Data drift detected in production traffic")
        
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Model Versioning and Updates

Implementing a system for model versioning and updates:

```python
import os
from datetime import datetime

class ModelManager:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.current_model = None
        self.current_tokenizer = None
        self.model_version = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load the latest model
        self.load_latest_model()
    
    def load_latest_model(self):
        # Find the latest model version
        versions = [d for d in os.listdir(self.model_dir) if os.path.isdir(os.path.join(self.model_dir, d))]
        if not versions:
            # No models found, load default
            self.load_model("distilbert-base-uncased-finetuned-sst-2-english")
            self.save_model("initial")
        else:
            # Load the latest version
            latest_version = max(versions)
            self.load_model_from_path(os.path.join(self.model_dir, latest_version))
            self.model_version = latest_version
    
    def load_model(self, model_name):
        self.current_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.current_model.to(device)
        self.current_model.eval()
        logger.info(f"Loaded model: {model_name}")
    
    def load_model_from_path(self, model_path):
        self.current_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.current_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.current_model.to(device)
        self.current_model.eval()
        logger.info(f"Loaded model from path: {model_path}")
    
    def save_model(self, version_name=None):
        if version_name is None:
            # Generate version name based on timestamp
            version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create version directory
        version_dir = os.path.join(self.model_dir, version_name)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.current_model.save_pretrained(version_dir)
        self.current_tokenizer.save_pretrained(version_dir)
        
        self.model_version = version_name
        logger.info(f"Saved model version: {version_name}")
        
        return version_name
    
    def update_model(self, new_model, new_tokenizer, version_name=None):
        # Save current model as backup
        backup_version = self.save_model(f"{self.model_version}_backup")
        
        # Update current model
        self.current_model = new_model
        self.current_tokenizer = new_tokenizer
        self.current_model.to(device)
        self.current_model.eval()
        
        # Save new model
        new_version = self.save_model(version_name)
        
        logger.info(f"Updated model to version: {new_version} (backup: {backup_version})")
        return new_version

# Create model manager
model_manager = ModelManager()

# Use model manager in prediction function
def predict_sentiment(text):
    inputs = model_manager.current_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model_manager.current_model(**inputs)
    
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    label_id = predictions.argmax().item()
    label = model_manager.current_model.config.id2label[label_id]
    score = predictions[label_id].item()
    
    return {"text": text, "label": label, "score": score}

# Add endpoint for model updates
class ModelUpdateRequest(BaseModel):
    model_path: str
    version_name: Optional[str] = None

@app.post("/update-model")
def update_model(request: ModelUpdateRequest):
    try:
        # Load new model
        new_model = AutoModelForSequenceClassification.from_pretrained(request.model_path)
        new_tokenizer = AutoTokenizer.from_pretrained(request.model_path)
        
        # Update model
        new_version = model_manager.update_model(new_model, new_tokenizer, request.version_name)
        
        return {"status": "success", "message": f"Model updated to version {new_version}"}
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## Looking Ahead

In this post, we've explored how to build production-ready applications with Hugging Face transformers. We've covered:

- Model serving strategies for batch and real-time inference
- Creating robust APIs with FastAPI
- Deployment options including containers and cloud platforms
- Monitoring and maintaining deployed models
- Handling data drift and model updates

These techniques form the foundation for deploying transformer models in production environments, ensuring they perform reliably and efficiently.

In our next post, we'll explore advanced topics and future directions in transformer models, including parameter-efficient fine-tuning, prompt engineering, emerging architectures, responsible AI considerations, and research frontiers.

Stay tuned for "Advanced Topics and Future Directions," where we'll discover cutting-edge techniques and the future of transformer models!

---

*This blog post is part of our "Mastering Hugging Face Transformers" series. Check back for new installments covering advanced topics, practical applications, and best practices.*