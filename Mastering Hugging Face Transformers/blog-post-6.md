# Multimodal Applications

Welcome to the sixth installment of our "Mastering Hugging Face Transformers" series! In our previous posts, we've explored transformer models, their architecture, pre-trained models, fine-tuning techniques, and specialized NLP tasks. Now, we're expanding beyond text to discover how transformers are revolutionizing multimodal applications.

In this post, we'll explore how transformer models can process images, audio, and combinations of different data types. We'll cover vision transformers, audio transformers, cross-modal models, and practical applications for multimodal processing.

## Vision Transformers (ViT)

### Understanding Vision Transformers

Vision Transformers (ViT) apply the transformer architecture to computer vision tasks. Unlike convolutional neural networks (CNNs), which have dominated computer vision for years, ViTs process images by:

1. Splitting images into fixed-size patches
2. Linearly embedding each patch
3. Adding position embeddings
4. Processing the resulting sequence with a standard transformer encoder

This approach has proven remarkably effective, with ViTs achieving state-of-the-art results on many vision tasks.

### How ViT Works

Here's how Vision Transformers process images:

```
Image → Split into patches → Linear embedding → Add position embeddings → Transformer encoder → Classification head
```

- **Patch extraction**: The image is divided into non-overlapping patches (typically 16×16 pixels)
- **Linear projection**: Each patch is flattened and projected to an embedding dimension
- **Position embedding**: Position information is added since transformers have no inherent sense of order
- **Transformer encoder**: Standard transformer blocks process the sequence of patch embeddings
- **Classification head**: A simple MLP on top of the [CLS] token output for classification tasks

### Implementing Vision Transformers

Let's implement an image classification solution using Vision Transformers:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
import torch
from PIL import Image

# Load a vision dataset (CIFAR-10)
dataset = load_dataset("cifar10")

# Examine the dataset
print(dataset)
print(f"Labels: {dataset['train'].features['label'].names}")

# Display an example image
image = dataset["train"][0]["img"]
print(f"Label: {dataset['train'].features['label'].names[dataset['train'][0]['label']]}")

# Load pre-trained ViT model and processor
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=10,
    id2label={str(i): c for i, c in enumerate(dataset["train"].features["label"].names)},
    label2id={c: str(i) for i, c in enumerate(dataset["train"].features["label"].names)}
)

# Prepare the dataset
def preprocess_images(examples):
    images = examples["img"]
    # Convert PIL images to RGB if they're not already
    images = [img.convert("RGB") for img in images]
    # Process images for the model
    inputs = processor(images, return_tensors="pt")
    return inputs

# Process a batch of images
processed_dataset = dataset.map(
    preprocess_images,
    batched=True,
    remove_columns=["img", "label"]
)

# Add labels
processed_dataset["train"] = processed_dataset["train"].add_column(
    "labels", dataset["train"]["label"]
)
processed_dataset["test"] = processed_dataset["test"].add_column(
    "labels", dataset["test"]["label"]
)

# Training setup
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"].select(range(5000)),  # Subset for demonstration
    eval_dataset=processed_dataset["test"].select(range(1000)),
    compute_metrics=compute_metrics,
)

# Train the model (commented out as it's resource-intensive)
# trainer.train()

# Inference with ViT
from transformers import pipeline

# Create an image classification pipeline
image_classifier = pipeline("image-classification", model=model, feature_extractor=processor)

# Test on new images
from PIL import Image
import requests
from io import BytesIO

# Function to load an image from URL
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Test images
test_image_urls = [
    "https://farm1.staticflickr.com/44/147426941_86f84e6573_z.jpg",  # a car
    "https://farm4.staticflickr.com/3224/3081748027_0ee3d59fea_z.jpg",  # a dog
    "https://farm8.staticflickr.com/7377/9359839441_b6509e1e27_z.jpg",  # a horse
]

for url in test_image_urls:
    image = load_image(url)
    result = image_classifier(image)
    print(f"Image: {url}")
    for r in result:
        print(f"Label: {r['label']}, Score: {r['score']:.4f}")
    print("---")
```

### Fine-tuning ViT for Custom Tasks

You can fine-tune ViT for specific vision tasks like object detection or segmentation:

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset

# Load a custom dataset
dataset = load_dataset("food101", split="train[:5000]")  # Using a subset

# Prepare the dataset
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

def preprocess_function(examples):
    images = [img.convert("RGB") for img in examples["image"]]
    examples["pixel_values"] = processor(images, return_tensors="pt")["pixel_values"]
    return examples

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=["image"]
)

# Load pre-trained model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=101,  # Food101 has 101 classes
    id2label={str(i): label for i, label in enumerate(dataset.features["label"].names)},
    label2id={label: str(i) for i, label in enumerate(dataset.features["label"].names)}
)

# Fine-tune the model
# (similar to the training setup above)
```

## Audio Transformers (Wav2Vec2, WhisperNet)

### Understanding Audio Transformers

Audio transformers apply the transformer architecture to speech and audio processing tasks. Models like Wav2Vec2 and Whisper have revolutionized speech recognition, audio classification, and other audio tasks.

These models typically:
1. Convert raw audio waveforms into representations
2. Process these representations with transformer encoders
3. Map the outputs to text (for speech recognition) or classes (for audio classification)

### Wav2Vec2 for Speech Recognition

Wav2Vec2 is a self-supervised model that learns representations from raw audio, making it particularly effective for speech recognition:

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import torch
import torchaudio

# Load a speech dataset
dataset = load_dataset("common_voice", "en", split="validation[:1000]")

# Load pre-trained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    # Load audio files
    audio_arrays = []
    for audio_file in examples["path"]:
        waveform, sample_rate = torchaudio.load(audio_file)
        # Convert to mono and resample if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        audio_arrays.append(waveform.squeeze().numpy())
    
    # Tokenize
    inputs = processor(
        audio_arrays,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # Tokenize targets
    with processor.as_target_processor():
        labels = processor(examples["sentence"], return_tensors="pt", padding=True)
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# Process the dataset
processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Training setup (similar to previous examples)

# Create a speech recognition pipeline
transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer)

# Test on a new audio file
from datasets import load_dataset
import soundfile as sf

# Load a test example
test_dataset = load_dataset("common_voice", "en", split="test[:1]")
audio_file = test_dataset[0]["path"]
audio_array, sampling_rate = sf.read(audio_file)

# Transcribe
transcription = transcriber(audio_array)
print(f"Audio file: {audio_file}")
print(f"Transcription: {transcription['text']}")
print(f"Reference: {test_dataset[0]['sentence']}")
```

### Whisper for Multilingual Speech Recognition

Whisper is a more recent model that excels at multilingual speech recognition and translation:

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load pre-trained Whisper model
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Function to transcribe audio
def transcribe_audio(audio_file):
    # Load and preprocess audio
    audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
    
    # Decode the generated ids
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Test on audio files in different languages
audio_files = {
    "english": "path/to/english_audio.wav",
    "french": "path/to/french_audio.wav",
    "spanish": "path/to/spanish_audio.wav"
}

for language, file in audio_files.items():
    transcription = transcribe_audio(file)
    print(f"Language: {language}")
    print(f"Transcription: {transcription}")
    print("---")

# Translate speech to English
def translate_speech_to_english(audio_file):
    # Load and preprocess audio
    audio_array, sampling_rate = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # Force the model to generate English
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
    
    # Generate translation
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids
        )
    
    # Decode the generated ids
    translation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translation

# Test translation
for language, file in audio_files.items():
    if language != "english":  # Only translate non-English files
        translation = translate_speech_to_english(file)
        print(f"Language: {language}")
        print(f"English translation: {translation}")
        print("---")
```

## Cross-modal Models (CLIP, DALL-E)

### Understanding Cross-modal Models

Cross-modal models can process and relate different types of data, such as text and images. These models enable powerful applications like:

- **Text-to-image retrieval**: Finding images that match a text description
- **Image-to-text retrieval**: Generating captions for images
- **Text-guided image generation**: Creating images from text descriptions

### CLIP: Connecting Text and Images

CLIP (Contrastive Language-Image Pre-training) learns visual concepts from natural language supervision, creating a unified space for text and images:

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch

# Load pre-trained CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Function to load an image from URL
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Test images
image_urls = [
    "https://farm4.staticflickr.com/3095/3140892215_7e62970d7b_z.jpg",  # dog
    "https://farm1.staticflickr.com/177/384835016_8c1e378d7e_z.jpg",    # cat
    "https://farm1.staticflickr.com/152/346365154_5f05d1fee1_z.jpg",    # car
    "https://farm3.staticflickr.com/2220/2478131890_07c22b2e28_z.jpg"    # bicycle
]

images = [load_image(url) for url in image_urls]

# Text descriptions
texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car",
    "a photo of a bicycle"
]

# Process inputs
inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

# Get similarity scores
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # image-text similarity score
    probs = logits_per_image.softmax(dim=1)      # probabilities

# Print results
print("Image-to-Text Similarity:")
for i, image_url in enumerate(image_urls):
    print(f"Image: {image_url}")
    for j, text in enumerate(texts):
        print(f"  {text}: {probs[i, j].item():.4f}")
    print("---")

# Zero-shot image classification with CLIP
def classify_image(image, candidate_labels):
    # Prepare text descriptions
    texts = [f"a photo of a {label}" for label in candidate_labels]
    
    # Process inputs
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    
    # Get similarity scores
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
    
    # Return results
    return {
        label: prob.item()
        for label, prob in zip(candidate_labels, probs)
    }

# Test zero-shot classification
test_image = load_image("https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z.jpg")
candidate_labels = ["dog", "cat", "horse", "elephant", "bird", "fish"]

results = classify_image(test_image, candidate_labels)
print("Zero-shot Classification Results:")
for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{label}: {score:.4f}")
```

### Text-to-Image Generation with Diffusion Models

Diffusion models like Stable Diffusion can generate images from text descriptions:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate images from text prompts
prompts = [
    "A serene landscape with mountains and a lake at sunset",
    "A futuristic city with flying cars and neon lights",
    "A photorealistic portrait of a smiling elderly woman",
    "An oil painting of a cat wearing a space suit"
]

for i, prompt in enumerate(prompts):
    # Generate image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(f"generated_image_{i}.png")
    
    print(f"Generated image for: {prompt}")
```

## Practical Applications for Multimodal Processing

### Visual Question Answering

Visual Question Answering (VQA) combines vision and language to answer questions about images:

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ViLT model
model_name = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_name)
model = ViltForQuestionAnswering.from_pretrained(model_name)

# Function to load an image from URL
def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Test image
image_url = "https://farm4.staticflickr.com/3075/3168662394_7d7103de7d_z.jpg"  # dog image
image = load_image(image_url)

# Questions about the image
questions = [
    "What animal is this?",
    "What color is the animal?",
    "Is the animal indoors or outdoors?",
    "Is the animal sleeping?"
]

for question in questions:
    # Prepare inputs
    inputs = processor(image, question, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get the predicted answer
    idx = logits.argmax(-1).item()
    print(f"Question: {question}")
    print(f"Answer: {model.config.id2label[idx]}")
    print("---")
```

### Image Captioning

Image captioning generates textual descriptions of images:

```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained image captioning model
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set special tokens
tokenizer.pad_token = tokenizer.eos_token

# Function to generate captions
def generate_caption(image):
    # Prepare image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    
    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=16,
            num_beams=4,
            return_dict_in_generate=True
        ).sequences
    
    # Decode the generated ids
    caption = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption

# Test on images
image_urls = [
    "https://farm4.staticflickr.com/3095/3140892215_7e62970d7b_z.jpg",  # dog
    "https://farm1.staticflickr.com/152/346365154_5f05d1fee1_z.jpg",    # car
    "https://farm3.staticflickr.com/2220/2478131890_07c22b2e28_z.jpg"    # bicycle
]

for url in image_urls:
    image = load_image(url)
    caption = generate_caption(image)
    print(f"Image: {url}")
    print(f"Caption: {caption}")
    print("---")
```

### Audio-Visual Speech Recognition

Combining audio and visual information can improve speech recognition, especially in noisy environments:

```python
from transformers import AutoProcessor, ASTForAudioClassification
import torch
import librosa
import numpy as np

# Load pre-trained audio model
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
processor = AutoProcessor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(model_name)

# Function to classify audio
def classify_audio(audio_file):
    # Load and preprocess audio
    waveform, sample_rate = librosa.load(audio_file, sr=16000)
    
    # Process audio
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get top predictions
    predicted_ids = torch.topk(logits, k=5).indices[0].tolist()
    predicted_labels = [model.config.id2label[id] for id in predicted_ids]
    scores = torch.softmax(logits, dim=1)[0][predicted_ids].tolist()
    
    return list(zip(predicted_labels, scores))

# Test on audio files
audio_file = "path/to/audio_sample.wav"
results = classify_audio(audio_file)

print(f"Audio: {audio_file}")
print("Classifications:")
for label, score in results:
    print(f"{label}: {score:.4f}")
```

## Looking Ahead

In this post, we've explored how transformer models have expanded beyond text to handle images, audio, and multimodal data. We've covered:

- Vision Transformers (ViT) for image processing
- Audio transformers like Wav2Vec2 and Whisper for speech recognition
- Cross-modal models like CLIP that connect text and images
- Practical applications including visual question answering and image captioning

These multimodal capabilities open up exciting possibilities for building more sophisticated AI systems that can understand and generate content across different modalities.

In our next post, we'll explore efficient transformers, focusing on techniques to make these powerful models more practical for real-world deployment. We'll cover model compression, quantization, knowledge distillation, and strategies for faster inference.

Stay tuned for "Efficient Transformers," where we'll learn how to make transformer models smaller, faster, and more deployable!

---

*This blog post is part of our "Mastering Hugging Face Transformers" series. Check back for new installments covering advanced topics, practical applications, and best practices.*