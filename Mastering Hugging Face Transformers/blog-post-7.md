# Efficient Transformers

Welcome to the seventh installment of our "Mastering Hugging Face Transformers" series! In our previous posts, we've explored transformer models, their architecture, pre-trained models, fine-tuning techniques, specialized NLP tasks, and multimodal applications. Now, we'll focus on making transformer models more efficient for practical deployment.

In this post, we'll explore techniques for model compression, quantization, knowledge distillation, faster inference, and running models on edge devices. These approaches are crucial for deploying transformer models in resource-constrained environments.

## Model Compression Techniques

### Understanding Model Compression

Transformer models are powerful but often very large, with models like GPT-3 containing 175 billion parameters. This size creates challenges for deployment:

- High memory requirements
- Slow inference times
- Excessive power consumption
- Limited deployability on edge devices

Model compression techniques address these issues by reducing model size while preserving performance.

### Pruning: Removing Unnecessary Weights

Pruning removes weights that contribute minimally to the model's performance. There are several pruning approaches:

- **Magnitude pruning**: Removing weights with small absolute values
- **Structured pruning**: Removing entire attention heads or layers
- **Movement pruning**: Removing weights that move toward zero during fine-tuning

Let's implement magnitude pruning for a BERT model:

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load a pre-trained model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Function to count non-zero parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Original model size: {count_parameters(model):,} parameters")

# Apply magnitude pruning
threshold = 0.1  # Pruning threshold

# Create a mask for each parameter
for name, param in model.named_parameters():
    if 'bias' not in name:  # Don't prune bias terms
        # Create a mask based on absolute value
        mask = (torch.abs(param) > threshold).float()
        # Apply the mask
        param.data.mul_(mask)

# Count non-zero parameters after pruning
def count_nonzero_parameters(model):
    return sum(torch.sum(p != 0).item() for p in model.parameters() if p.requires_grad)

print(f"Pruned model size: {count_nonzero_parameters(model):,} non-zero parameters")
print(f"Sparsity: {1 - count_nonzero_parameters(model) / count_parameters(model):.2%}")

# Save the pruned model
model.save_pretrained("./pruned-bert")
```

### Structured Pruning with Hugging Face

Hugging Face provides tools for more sophisticated pruning:

```python
from transformers import BertForSequenceClassification, BertConfig
from transformers.pruning import prune_linear_layer

# Load a pre-trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Prune specific attention heads
heads_to_prune = {0: [0, 2], 1: [4, 5, 6]}  # Layer_index: list of heads to prune

for layer, heads in heads_to_prune.items():
    # Prune attention heads in the encoder
    model.bert.encoder.layer[layer].attention.prune_heads(heads)

print(f"Model size after pruning attention heads: {count_parameters(model):,} parameters")
```

## Quantization and Pruning

### Understanding Quantization

Quantization reduces the precision of model weights, typically from 32-bit floating-point to 8-bit integers or even lower. This significantly reduces memory usage and can speed up inference, especially on hardware with integer acceleration.

### Post-Training Quantization

Post-training quantization (PTQ) applies quantization after a model has been trained:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a pre-trained model
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Original model size
original_size = model.state_dict()['bert.encoder.layer.0.attention.self.query.weight'].element_size() * count_parameters(model)
print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")

# Quantize the model to INT8
quantized_model = torch.quantization.quantize_dynamic(
    model,  # Model to quantize
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8  # Target dtype
)

# Quantized model size (approximate)
quantized_size = original_size / 4  # 32-bit to 8-bit is a 4x reduction
print(f"Quantized model size (approximate): {quantized_size / 1024 / 1024:.2f} MB")

# Test the quantized model
inputs = tokenizer("This is a test sentence", return_tensors="pt")
with torch.no_grad():
    outputs = quantized_model(**inputs)

print("Quantized model output:", outputs.logits)
```

### Quantization-Aware Training

Quantization-aware training (QAT) simulates quantization during training to improve accuracy:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load a dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare the model for quantization-aware training
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define a custom QAT model
class QuantizedBERT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Prepare for QAT
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, **inputs):
        # Quantize inputs
        quant_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float:
                quant_inputs[k] = self.quant(v)
            else:
                quant_inputs[k] = v
        
        # Forward pass
        outputs = self.model(**quant_inputs)
        
        # Dequantize outputs
        if isinstance(outputs.logits, torch.Tensor):
            outputs.logits = self.dequant(outputs.logits)
        
        return outputs

# Create the QAT model
qat_model = QuantizedBERT(model)

# Configure quantization
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# Prepare the model for QAT
torch.quantization.prepare_qat(qat_model, inplace=True)

# Train with QAT
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=qat_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Convert to a quantized model
quantized_model = torch.quantization.convert(qat_model, inplace=False)

# Save the quantized model
torch.save(quantized_model.state_dict(), "quantized_bert.pth")
```

### Combining Pruning and Quantization

Combining pruning and quantization can yield even greater compression:

```python
# First prune the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Apply magnitude pruning (as shown earlier)
threshold = 0.1
for name, param in model.named_parameters():
    if 'bias' not in name:
        mask = (torch.abs(param) > threshold).float()
        param.data.mul_(mask)

# Then quantize the pruned model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Evaluate the compressed model
# ...
```

## Knowledge Distillation

### Understanding Knowledge Distillation

Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model. The student learns not just from ground truth labels but also from the teacher's probability distributions, which contain rich information about relationships between classes.

### Implementing Knowledge Distillation

Let's implement knowledge distillation from BERT to a smaller model:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn.functional as F

# Load dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load teacher model (BERT)
teacher_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Load student model (DistilBERT)
student_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Generate teacher predictions
def get_teacher_predictions(dataset):
    teacher_outputs = []
    teacher_model.eval()
    
    for i in range(0, len(dataset), 32):
        batch = dataset[i:i+32]
        inputs = {k: torch.tensor(v) for k, v in batch.items() if k != "label"}
        with torch.no_grad():
            outputs = teacher_model(**inputs)
        teacher_outputs.extend(outputs.logits.detach())
    
    return torch.stack(teacher_outputs)

# Get teacher predictions for training set
teacher_preds = get_teacher_predictions(tokenized_datasets["train"])

# Custom distillation trainer
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_preds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_preds = teacher_preds
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student predictions
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Standard cross-entropy loss
        labels = inputs.pop("labels")
        ce_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss
        batch_indices = inputs["input_ids"].shape[0]  # Batch size
        teacher_batch = self.teacher_preds[self.state.global_step * batch_indices:(self.state.global_step + 1) * batch_indices]
        
        # Temperature-scaled distillation loss
        temperature = 2.0
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_batch / temperature, dim=-1),
            reduction="batchmean"
        ) * (temperature ** 2)
        
        # Combined loss
        alpha = 0.5  # Weight for distillation loss
        loss = alpha * distillation_loss + (1 - alpha) * ce_loss
        
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir="./distilled_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Create distillation trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    teacher_preds=teacher_preds,
)

# Train the student model
trainer.train()

# Save the distilled model
trainer.save_model("./distilled_model")

# Compare model sizes
print(f"Teacher model size: {count_parameters(teacher_model):,} parameters")
print(f"Student model size: {count_parameters(student_model):,} parameters")
print(f"Size reduction: {1 - count_parameters(student_model) / count_parameters(teacher_model):.2%}")
```

### Using Pre-Distilled Models

Hugging Face provides pre-distilled models that you can use directly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load a pre-distilled model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test the model
texts = [
    "I absolutely loved this movie! The acting was superb.",
    "What a waste of time. The plot made no sense at all.",
    "It was okay, not great but not terrible either."
]

results = sentiment_analyzer(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
    print("---")
```

## Faster Inference with ONNX and TensorRT

### Understanding ONNX

ONNX (Open Neural Network Exchange) is an open format for representing machine learning models. Converting models to ONNX enables deployment across different frameworks and hardware accelerators.

### Converting Transformers to ONNX

Let's convert a Hugging Face model to ONNX:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.onnx import export
from pathlib import Path

# Load model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create output directory
output_path = Path("./onnx_model")
output_path.mkdir(exist_ok=True)

# Export to ONNX
export(
    preprocessor=tokenizer,
    model=model,
    output=output_path / "model.onnx",
    opset=12,  # ONNX opset version
)

print(f"Model exported to: {output_path / 'model.onnx'}")

# Inference with ONNX Runtime
import onnxruntime as ort
import numpy as np

# Create ONNX Runtime session
session = ort.InferenceSession(str(output_path / "model.onnx"))

# Prepare input
text = "This movie was fantastic!"
inputs = tokenizer(text, return_tensors="np")

# Run inference
onnx_inputs = {k: v for k, v in inputs.items()}
outputs = session.run(None, onnx_inputs)

# Process outputs
logits = outputs[0]
predicted_class = np.argmax(logits, axis=1)[0]
print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

### Optimizing with TensorRT

TensorRT is a high-performance deep learning inference optimizer for NVIDIA GPUs:

```python
# Note: This requires TensorRT to be installed
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load ONNX model and build TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("./onnx_model/model.onnx", "rb") as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)

# Save engine
with open("./tensorrt_model.engine", "wb") as f:
    f.write(engine.serialize())

print("TensorRT engine built and saved.")

# Inference with TensorRT
context = engine.create_execution_context()

# Prepare input
text = "This movie was fantastic!"
inputs = tokenizer(text, return_tensors="np")

# Allocate buffers
binding_idxs = {}
input_buffers = {}
output_buffers = {}

for i in range(engine.num_bindings):
    binding_name = engine.get_binding_name(i)
    binding_dims = engine.get_binding_shape(i)
    binding_size = trt.volume(binding_dims) * np.dtype(np.float32).itemsize
    
    binding_idxs[binding_name] = i
    
    # Allocate GPU memory
    device_buffer = cuda.mem_alloc(binding_size)
    
    if engine.binding_is_input(i):
        input_buffers[binding_name] = device_buffer
    else:
        output_buffers[binding_name] = (device_buffer, binding_dims)

# Copy input data to GPU
for name, buffer in input_buffers.items():
    # Get corresponding input from tokenizer
    input_data = inputs[name].astype(np.float32)
    # Copy to GPU
    cuda.memcpy_htod(buffer, input_data)

# Run inference
context.execute_v2(list(input_buffers.values()) + [out[0] for out in output_buffers.values()])

# Copy output from GPU to CPU
output_data = {}
for name, (buffer, shape) in output_buffers.items():
    output = np.zeros(shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, buffer)
    output_data[name] = output

# Process output
logits = output_data["logits"]
predicted_class = np.argmax(logits, axis=1)[0]
print(f"TensorRT predicted class: {model.config.id2label[predicted_class]}")
```

## Running Models on Edge Devices

### Optimizing for Mobile Devices

To deploy transformer models on mobile devices, you can use frameworks like TensorFlow Lite or PyTorch Mobile:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load a small model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save tokenizer for mobile
tokenizer.save_pretrained("./mobile_model")

# Optimize model for mobile
model.eval()  # Set to evaluation mode

# Create example input for tracing
example_text = "This is an example sentence."
example_inputs = tokenizer(example_text, return_tensors="pt")

# Trace the model with example inputs
traced_model = torch.jit.trace(
    model,
    (example_inputs["input_ids"], example_inputs["attention_mask"]),
)

# Optimize the traced model
optimized_model = torch.jit.optimize_for_mobile(traced_model)

# Save the optimized model
optimized_model._save_for_lite_interpreter("./mobile_model/model.ptl")
print("Model optimized and saved for mobile deployment.")
```

### Deploying with Hugging Face Optimum

Hugging Face Optimum provides tools for optimizing and deploying models on various hardware:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Load model directly as an optimized ONNX model
ort_model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    from_transformers=True,
)

# Save the optimized model
ort_model.save_pretrained("./optimized_model")

# Inference
inputs = tokenizer("This is a test", return_tensors="pt")
outputs = ort_model(**inputs)

print("Prediction:", outputs.logits.argmax(-1).item())
```

### Quantized Models for Edge Devices

For extremely resource-constrained devices, you can use int8 or even int4 quantization:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Quantize to 8-bit
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save the quantized model
torch.save(quantized_model.state_dict(), "./edge_model/model_int8.pth")

# For extreme quantization (4-bit), specialized libraries like GPTQ or bitsandbytes can be used
# Example with bitsandbytes (requires installation)
# from transformers import AutoModelForSequenceClassification
# model = AutoModelForSequenceClassification.from_pretrained(
#     "distilbert-base-uncased-finetuned-sst-2-english",
#     load_in_4bit=True,
#     quantization_config={"bnb_4bit_compute_dtype": torch.float16}
# )
```

## Looking Ahead

In this post, we've explored various techniques to make transformer models more efficient and deployable:

- Model compression through pruning to remove unnecessary weights
- Quantization to reduce precision and memory requirements
- Knowledge distillation to create smaller, faster models
- Optimization with ONNX and TensorRT for faster inference
- Strategies for deploying models on edge devices

These techniques are essential for bringing the power of transformer models to real-world applications, especially in resource-constrained environments.

In our next post, we'll explore building production-ready applications with Hugging Face transformers. We'll cover model serving strategies, creating APIs, deployment options, monitoring, and maintaining deployed models.

Stay tuned for "Building Production-Ready Applications," where we'll learn how to take transformer models from research to production!

---

*This blog post is part of our "Mastering Hugging Face Transformers" series. Check back for new installments covering advanced topics, practical applications, and best practices.*