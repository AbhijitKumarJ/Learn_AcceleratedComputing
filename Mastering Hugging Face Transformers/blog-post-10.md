# Practical Project Walkthrough

Welcome to the final installment of our "Mastering Hugging Face Transformers" series! Throughout our previous posts, we've explored transformer models, their architecture, pre-trained models, fine-tuning techniques, specialized NLP tasks, multimodal applications, efficient transformers, production deployment, and advanced topics. Now, we'll bring everything together with a practical end-to-end project.

In this post, we'll build a complete application that demonstrates many of the concepts we've covered. Our project will be a multilingual document processing system that can analyze, summarize, translate, and answer questions about documents in different languages.

## Project Overview

### The Challenge

Many organizations deal with documents in multiple languages and need to extract insights, generate summaries, and answer specific questions about their content. Our application will address this need by providing a comprehensive document processing pipeline.

### Features

Our application will include the following features:

1. **Document understanding**: Extract key information from documents
2. **Multilingual support**: Process documents in different languages
3. **Summarization**: Generate concise summaries of documents
4. **Translation**: Translate documents to the user's preferred language
5. **Question answering**: Answer specific questions about document content
6. **Efficient processing**: Optimize for performance and resource usage
7. **API and web interface**: Make the system accessible via API and web UI

### Architecture

The application will have the following components:

1. **Document processor**: Handles document parsing and preprocessing
2. **Language detector**: Identifies the document's language
3. **Information extractor**: Extracts key entities and information
4. **Summarizer**: Generates document summaries
5. **Translator**: Translates content between languages
6. **Question answering engine**: Answers questions about the document
7. **API layer**: Provides access to all features via REST API
8. **Web interface**: Offers a user-friendly front end

## Implementation

### Setting Up the Project

Let's start by setting up our project structure and installing dependencies:

```python
# app.py - Main application file

import os
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import time
import uuid
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("document-processor")

# Create FastAPI app
app = FastAPI(title="Multilingual Document Processor", 
              description="Process, summarize, translate, and query documents in multiple languages")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
upload_dir = Path("./uploads")
upload_dir.mkdir(exist_ok=True)

# Create results directory
results_dir = Path("./results")
results_dir.mkdir(exist_ok=True)

# Store for background jobs
job_store = {}

# Define request and response models
class ProcessRequest(BaseModel):
    target_language: str = "english"
    generate_summary: bool = True
    extract_entities: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "target_language": "english",
                "generate_summary": True,
                "extract_entities": True
            }
        }

class QuestionRequest(BaseModel):
    document_id: str
    question: str
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_12345",
                "question": "What are the main points discussed in the document?"
            }
        }

class JobResponse(BaseModel):
    job_id: str
    status: str

class ProcessingResult(BaseModel):
    document_id: str
    original_language: str
    translated_text: Optional[str] = None
    summary: Optional[str] = None
    entities: Optional[Dict[str, List[str]]] = None
    processing_time: float

class QuestionResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    confidence: float

# Mount static files for web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define API endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Multilingual Document Processor API"}

# We'll implement the rest of the endpoints as we build each component
```

### Document Processor Component

Let's implement the document processor to handle different file types:

```python
# document_processor.py

import PyPDF2
import docx
import re
from pathlib import Path
import logging

logger = logging.getLogger("document-processor")

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = {
            ".txt": self._process_txt,
            ".pdf": self._process_pdf,
            ".docx": self._process_docx
        }
    
    def process_document(self, file_path):
        """Process a document and extract its text content."""
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            processor = self.supported_formats[file_extension]
            text = processor(file_path)
            
            # Clean the text
            text = self._clean_text(text)
            
            return text
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _process_txt(self, file_path):
        """Extract text from a .txt file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    
    def _process_pdf(self, file_path):
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return text
    
    def _process_docx(self, file_path):
        """Extract text from a .docx file."""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _clean_text(self, text):
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters and formatting artifacts
        text = re.sub(r'[^\w\s.,;:!?()\[\]{}"\'-]', '', text)
        return text.strip()
```

### Language Detection and Translation

Next, let's implement language detection and translation using Hugging Face models:

```python
# language_processor.py

from transformers import MarianMTModel, MarianTokenizer, pipeline
import torch
import logging

logger = logging.getLogger("document-processor")

class LanguageProcessor:
    def __init__(self):
        # Initialize language detection pipeline
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        
        # Map of language codes to full names
        self.language_map = {
            "en": "english",
            "fr": "french",
            "de": "german",
            "es": "spanish",
            "it": "italian",
            "pt": "portuguese",
            "nl": "dutch",
            "ru": "russian",
            "zh": "chinese",
            "ja": "japanese",
            "ar": "arabic"
        }
        
        # Reverse mapping
        self.language_code_map = {v: k for k, v in self.language_map.items()}
        
        # Translation model cache
        self.translation_models = {}
        self.translation_tokenizers = {}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def detect_language(self, text):
        """Detect the language of the given text."""
        try:
            # Use a sample of the text for faster detection
            sample = text[:1000]
            result = self.language_detector(sample)[0]
            language_code = result["label"]
            confidence = result["score"]
            
            # Map language code to full name
            language = self.language_map.get(language_code, language_code)
            
            logger.info(f"Detected language: {language} with confidence {confidence:.4f}")
            return language, confidence
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return "english", 0.0  # Default to English on error
    
    def translate(self, text, source_language, target_language):
        """Translate text from source language to target language."""
        try:
            # If source and target are the same, return original text
            if source_language.lower() == target_language.lower():
                return text
            
            # Get language codes
            source_code = self.language_code_map.get(source_language.lower(), source_language.lower())
            target_code = self.language_code_map.get(target_language.lower(), target_language.lower())
            
            # Create model key
            model_key = f"{source_code}-{target_code}"
            
            # Load translation model if not already loaded
            if model_key not in self.translation_models:
                # For Helsinki-NLP models
                model_name = f"Helsinki-NLP/opus-mt-{source_code}-{target_code}"
                try:
                    self.translation_tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
                    self.translation_models[model_key] = MarianMTModel.from_pretrained(model_name).to(self.device)
                except Exception as e:
                    # Try the reverse direction with special token
                    logger.warning(f"Could not load direct model: {str(e)}")
                    try:
                        model_name = "facebook/m2m100_418M"  # Multilingual model
                        self.translation_tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
                        self.translation_models[model_key] = MarianMTModel.from_pretrained(model_name).to(self.device)
                    except Exception as e2:
                        logger.error(f"Could not load translation model: {str(e2)}")
                        return text  # Return original text on error
            
            # Split text into chunks to avoid GPU memory issues
            max_chunk_length = 512
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            translated_chunks = []
            
            for chunk in chunks:
                # Tokenize
                tokenizer = self.translation_tokenizers[model_key]
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_chunk_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Translate
                with torch.no_grad():
                    outputs = self.translation_models[model_key].generate(**inputs)
                
                # Decode
                translated_chunk = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                translated_chunks.append(translated_chunk)
            
            # Combine chunks
            translated_text = " ".join(translated_chunks)
            
            return translated_text
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return text  # Return original text on error
```

### Information Extraction

Let's implement entity extraction using a named entity recognition model:

```python
# information_extractor.py

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
import logging

logger = logging.getLogger("document-processor")

class InformationExtractor:
    def __init__(self):
        # Initialize NER pipeline
        self.ner_model_name = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(self.ner_model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.ner_model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"Information extractor initialized on {self.device}")
    
    def extract_entities(self, text):
        """Extract named entities from text."""
        try:
            # Process text in chunks to avoid memory issues
            max_chunk_length = 512
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            all_entities = []
            for chunk in chunks:
                entities = self.ner_pipeline(chunk)
                all_entities.extend(entities)
            
            # Group entities by type
            grouped_entities = {}
            for entity in all_entities:
                entity_type = entity["entity_group"]
                entity_text = entity["word"]
                
                if entity_type not in grouped_entities:
                    grouped_entities[entity_type] = []
                
                # Avoid duplicates
                if entity_text not in grouped_entities[entity_type]:
                    grouped_entities[entity_type].append(entity_text)
            
            return grouped_entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {}  # Return empty dict on error
```

### Document Summarization

Next, let's implement the summarization component:

```python
# summarizer.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import logging

logger = logging.getLogger("document-processor")

class Summarizer:
    def __init__(self):
        # Initialize summarization pipeline
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create pipeline
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=0 if self.device.type == "cuda" else -1)
        
        logger.info(f"Summarizer initialized on {self.device}")
    
    def summarize(self, text, max_length=150, min_length=50):
        """Generate a summary of the given text."""
        try:
            # If text is too short, return it as is
            if len(text.split()) < min_length:
                return text
            
            # Split text into chunks if it's too long
            max_input_length = 1024  # BART's maximum input length
            chunks = self._split_into_chunks(text, max_input_length)
            
            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length // len(chunks),
                    min_length=min_length // len(chunks),
                    do_sample=False
                )[0]["summary_text"]
                summaries.append(summary)
            
            # Combine summaries
            combined_summary = " ".join(summaries)
            
            # If the combined summary is still too long, summarize it again
            if len(combined_summary.split()) > max_length and len(chunks) > 1:
                combined_summary = self.summarizer(
                    combined_summary,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]["summary_text"]
            
            return combined_summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."  # Return error message
    
    def _split_into_chunks(self, text, max_chunk_length):
        """Split text into chunks based on token length."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Approximate token length (words are usually shorter than tokens)
            word_token_length = len(self.tokenizer.tokenize(word))
            
            if current_length + word_token_length > max_chunk_length:
                # Current chunk is full, start a new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_token_length
            else:
                # Add word to current chunk
                current_chunk.append(word)
                current_length += word_token_length
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
```

### Question Answering Engine

Let's implement the question answering component:

```python
# qa_engine.py

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch
import logging

logger = logging.getLogger("document-processor")

class QuestionAnsweringEngine:
    def __init__(self):
        # Initialize QA pipeline
        self.model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create pipeline
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer, device=0 if self.device.type == "cuda" else -1)
        
        logger.info(f"QA engine initialized on {self.device}")
    
    def answer_question(self, question, context):
        """Answer a question based on the given context."""
        try:
            # If context is too long, we need to split it and find the most relevant chunk
            max_context_length = 512  # Maximum context length for the model
            if len(self.tokenizer.tokenize(context)) > max_context_length:
                chunks = self._split_context(context, max_context_length)
                return self._answer_with_chunks(question, chunks)
            else:
                # Context fits within model's limits
                result = self.qa_pipeline(question=question, context=context)
                return result["answer"], result["score"]
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "Unable to answer the question.", 0.0
    
    def _split_context(self, context, max_length):
        """Split context into overlapping chunks."""
        words = context.split()
        chunks = []
        chunk_size = max_length - 50  # Leave room for question and overlap
        overlap = 50  # Number of overlapping words between chunks
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _answer_with_chunks(self, question, chunks):
        """Find answer in multiple chunks and return the best one."""
        best_answer = ""
        best_score = 0
        
        for chunk in chunks:
            result = self.qa_pipeline(question=question, context=chunk)
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]
        
        return best_answer, best_score
```

### Putting It All Together

Now, let's integrate all components into our main application:

```python
# Add these imports to app.py
from document_processor import DocumentProcessor
from language_processor import LanguageProcessor
from information_extractor import InformationExtractor
from summarizer import Summarizer
from qa_engine import QuestionAnsweringEngine
import shutil

# Initialize components
document_processor = DocumentProcessor()
language_processor = LanguageProcessor()
information_extractor = InformationExtractor()
summarizer = Summarizer()
qa_engine = QuestionAnsweringEngine()

# Document cache
document_cache = {}

# Process document function
def process_document(file_path, target_language, generate_summary, extract_entities):
    """Process a document and return results."""
    try:
        start_time = time.time()
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        # Extract text from document
        text = document_processor.process_document(file_path)
        
        # Detect language
        original_language, _ = language_processor.detect_language(text)
        
        # Initialize result dictionary
        result = {
            "document_id": document_id,
            "original_language": original_language,
            "original_text": text
        }
        
        # Translate if needed
        if target_language.lower() != original_language.lower():
            translated_text = language_processor.translate(text, original_language, target_language)
            result["translated_text"] = translated_text
        else:
            result["translated_text"] = text  # Same language, no translation needed
        
        # Generate summary if requested
        if generate_summary:
            # Summarize in the target language
            summary = summarizer.summarize(result["translated_text"])
            result["summary"] = summary
        
        # Extract entities if requested
        if extract_entities:
            # Extract from the target language text
            entities = information_extractor.extract_entities(result["translated_text"])
            result["entities"] = entities
        
        # Calculate processing time
        end_time = time.time()
        result["processing_time"] = end_time - start_time
        
        # Cache the document for question answering
        document_cache[document_id] = result
        
        return result
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

# Background processing task
def process_document_task(job_id, file_path, target_language, generate_summary, extract_entities):
    try:
        # Update job status
        job_store[job_id]["status"] = "processing"
        
        # Process document
        result = process_document(file_path, target_language, generate_summary, extract_entities)
        
        # Update job with results
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = result
    except Exception as e:
        # Update job with error
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {str(e)}")

# Add API endpoints
@app.post("/upload", response_model=JobResponse)
async def upload_document(
    file: UploadFile = File(...),
    target_language: str = Form("english"),
    generate_summary: bool = Form(True),
    extract_entities: bool = Form(True),
    background_tasks: BackgroundTasks = None
):
    try:
        # Generate job ID
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded file
        file_path = upload_dir / f"{job_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create job in store
        job_store[job_id] = {
            "status": "queued",
            "file_path": str(file_path),
            "filename": file.filename,
            "target_language": target_language,
            "generate_summary": generate_summary,
            "extract_entities": extract_entities,
            "created_at": time.time()
        }
        
        # Process document in background
        background_tasks.add_task(
            process_document_task,
            job_id,
            file_path,
            target_language,
            generate_summary,
            extract_entities
        )
        
        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_store[job_id]
    
    if job["status"] == "completed":
        return {
            "status": job["status"],
            "result": job["result"]
        }
    elif job["status"] == "failed":
        return {"status": job["status"], "error": job["error"]}
    else:
        return {"status": job["status"]}

@app.post("/question", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    try:
        document_id = request.document_id
        question = request.question
        
        # Check if document exists in cache
        if document_id not in document_cache:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document from cache
        document = document_cache[document_id]
        
        # Use translated text as context
        context = document["translated_text"]
        
        # Get answer from QA engine
        answer, confidence = qa_engine.answer_question(question, context)
        
        return {
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Creating a Simple Web Interface

Let's create a basic web interface for our application:

```html
<!-- static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual Document Processor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 2rem; }
        .result-section { margin-top: 2rem; display: none; }
        .loading { display: none; }
        .entity-tag { display: inline-block; margin: 0.2rem; padding: 0.2rem 0.5rem; border-radius: 0.25rem; }
        .entity-PER { background-color: #ffcccc; }
        .entity-ORG { background-color: #ccffcc; }
        .entity-LOC { background-color: #ccccff; }
        .entity-MISC { background-color: #ffffcc; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Multilingual Document Processor</h1>
        
        <div class="card mb-4">
            <div class="card-header">Upload Document</div>
            <div class="card-body">
                <form id="upload-form">
                    <div class="mb-3">
                        <label for="document" class="form-label">Document</label>
                        <input type="file" class="form-control" id="document" name="file" required>
                        <div class="form-text">Supported formats: PDF, DOCX, TXT</div>
                    </div>
                    
                    <div class="mb-3">
                        <label for="target-language" class="form-label">Target Language</label>
                        <select class="form-select" id="target-language" name="target_language">
                            <option value="english" selected>English</option>
                            <option value="french">French</option>
                            <option value="german">German</option>
                            <option value="spanish">Spanish</option>
                            <option value="italian">Italian</option>
                            <option value="portuguese">Portuguese</option>
                            <option value="dutch">Dutch</option>
                            <option value="russian">Russian</option>
                            <option value="chinese">Chinese</option>
                            <option value="japanese">Japanese</option>
                            <option value="arabic">Arabic</option>
                        </select>
                    </div>
                    
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="generate-summary" name="generate_summary" checked>
                        <label class="form-check-label" for="generate-summary">Generate Summary</label>
                    </div>
                    
                    <div class="mb-3 form-check">
                        <input type="checkbox" class="form-check-input" id="extract-entities" name="extract_entities" checked>
                        <label class="form-check-label" for="extract-entities">Extract Entities</label>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Process Document</button>
                </form>
                
                <div class="loading mt-3">
                    <div class="d-flex align-items-center">
                        <div class="spinner-border text-primary me-2" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <div>Processing document... This may take a few minutes.</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="result-section" class="result-section">
            <h2>Results</h2>
            
            <div class="card mb-4">
                <div class="card-header">Document Information</div>
                <div class="card-body">
                    <p><strong>Document ID:</strong> <span id="document-id"></span></p>
                    <p><strong>Original Language:</strong> <span id="original-language"></span></p>
                    <p><strong>Processing Time:</strong> <span id="processing-time"></span> seconds</p>
                </div>
            </div>
            
            <div id="summary-card" class="card mb-4">
                <div class="card-header">Summary</div>
                <div class="card-body">
                    <p id="summary-text"></p>
                </div>
            </div>
            
            <div id="entities-card" class="card mb-4">
                <div class="card-header">Extracted Entities</div>
                <div class="card-body">
                    <div id="entities-container"></div>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">Ask a Question</div>
                <div class="card-body">
                    <form id="question-form">
                        <div class="mb-3">
                            <label for="question" class="form-label">Question</label>
                            <input type="text" class="form-control" id="question" name="question" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Ask</button>
                    </form>
                    
                    <div id="answer-container" class="mt-3" style="display: none;">
                        <h5>Answer:</h5>
                        <p id="answer-text"></p>
                        <p><small>Confidence: <span id="answer-confidence"></span></small></p>
                    </div>
                </div>
            </div>
            
            <div class="card mb-4">
                <div class="card-header">Full Text</div>
                <div class="card-body">
                    <p id="full-text"></p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Document processing
        const uploadForm = document.getElementById('upload-form');
        const loading = document.querySelector('.loading');
        const resultSection = document.getElementById('result-section');
        let currentDocumentId = null;
        
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            loading.style.display = 'block';
            resultSection.style.display = 'none';
            
            const formData = new FormData(uploadForm);
            
            try {
                // Upload document
                const uploadResponse = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!uploadResponse.ok) {
                    throw new Error('Failed to upload document');
                }
                
                const uploadData = await uploadResponse.json();
                const jobId = uploadData.job_id;
                
                // Poll for job completion
                await pollJobStatus(jobId);
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing document: ' + error.message);
                loading.style.display = 'none';
            }
        });
        
        async function pollJobStatus(jobId) {
            try {
                const response = await fetch(`/job/${jobId}`);
                const data = await response.json();
                
                if (data.status === 'completed') {
                    // Job completed successfully
                    displayResults(data.result);
                    loading.style.display = 'none';
                    resultSection.style.display = 'block';
                } else if (data.status === 'failed') {
                    // Job failed
                    throw new Error(data.error || 'Processing failed');
                } else {
                    // Job still in progress, poll again after 2 seconds
                    setTimeout(() => pollJobStatus(jobId), 2000);
                }
            } catch (error) {
                console.error('Error polling job status:', error);
                alert('Error processing document: ' + error.message);
                loading.style.display = 'none';
            }
        }
        
        function displayResults(result) {
            // Store document ID for question answering
            currentDocumentId = result.document_id;
            
            // Display document information
            document.getElementById('document-id').textContent = result.document_id;
            document.getElementById('original-language').textContent = result.original_language;
            document.getElementById('processing-time').textContent = result.processing_time.toFixed(2);
            
            // Display summary if available
            const summaryCard = document.getElementById('summary-card');
            if (result.summary) {
                document.getElementById('summary-text').textContent = result.summary;
                summaryCard.style.display = 'block';
            } else {
                summaryCard.style.display = 'none';
            }
            
            // Display entities if available
            const entitiesCard = document.getElementById('entities-card');
            const entitiesContainer = document.getElementById('entities-container');
            if (result.entities && Object.keys(result.entities).length > 0) {
                entitiesContainer.innerHTML = '';
                
                for (const [entityType, entities] of Object.entries(result.entities)) {
                    const typeHeading = document.createElement('h5');
                    typeHeading.textContent = entityType;
                    entitiesContainer.appendChild(typeHeading);
                    
                    const entityList = document.createElement('div');
                    entityList.className = 'mb-3';
                    
                    for (const entity of entities) {
                        const entityTag = document.createElement('span');
                        entityTag.className = `entity-tag entity-${entityType}`;
                        entityTag.textContent = entity;
                        entityList.appendChild(entityTag);
                    }
                    
                    entitiesContainer.appendChild(entityList);
                }
                
                entitiesCard.style.display = 'block';
            } else {
                entitiesCard.style.display = 'none';
            }
            
            // Display full text
            document.getElementById('full-text').textContent = result.translated_text || result.original_text;
        }
        
        // Question answering
        const questionForm = document.getElementById('question-form');
        const answerContainer = document.getElementById('answer-container');
        
        questionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!currentDocumentId) {
                alert('No document has been processed yet.');
                return;
            }
            
            const question = document.getElementById('question').value;
            
            try {
                const response = await fetch('/question', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        document_id: currentDocumentId,
                        question: question
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to get answer');
                }
                
                const data = await response.json();
                
                // Display answer
                document.getElementById('answer-text').textContent = data.answer;
                document.getElementById('answer-confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
                answerContainer.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('Error getting answer: ' + error.message);
            }
        });
    </script>
</body>
</html>
```

### Adding the API Endpoint for the Web Interface

Add this endpoint to app.py to serve the web interface:

```python
from fastapi.responses import HTMLResponse

@app.get("/ui", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r") as f:
        return f.read()
```

## Optimizations and Best Practices

Let's implement some optimizations and best practices for our application:

### Model Caching and Lazy Loading

To optimize memory usage, we can implement lazy loading for our models:

```python
# Add this to language_processor.py

class LazyModel:
    def __init__(self, model_class, model_name, device):
        self.model_class = model_class
        self.model_name = model_name
        self.device = device
        self.model = None
    
    def __call__(self, *args, **kwargs):
        if self.model is None:
            logger.info(f"Lazy loading model: {self.model_name}")
            self.model = self.model_class.from_pretrained(self.model_name).to(self.device)
        return self.model(*args, **kwargs)

# Then modify the LanguageProcessor initialization
self.translation_models[model_key] = LazyModel(MarianMTModel, model_name, self.device)
```

### Batch Processing for Efficiency

Implement batch processing for more efficient inference:

```python
# Add this to information_extractor.py

def extract_entities_batch(self, texts):
    """Extract entities from multiple texts in a batch."""
    try:
        all_entities = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process outputs for each text
            # ...
        
        return all_entities
    except Exception as e:
        logger.error(f"Error batch extracting entities: {str(e)}")
        return [{}] * len(texts)
```

### Error Handling and Graceful Degradation

Implement more robust error handling:

```python
# Add this to app.py

class DocumentProcessingError(Exception):
    """Exception raised for errors in document processing."""
    pass

# Modify process_document function
def process_document(file_path, target_language, generate_summary, extract_entities):
    """Process a document and return results."""
    try:
        # ... existing code ...
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Attempt partial processing
        try:
            # If we at least extracted the text, return that
            if 'text' in locals():
                result = {
                    "document_id": document_id,
                    "original_language": "unknown",
                    "original_text": text,
                    "error": str(e),
                    "partial_result": True
                }
                return result
        except:
            pass
        
        # If all else fails, raise the error
        raise DocumentProcessingError(f"Failed to process document: {str(e)}")
```

### Monitoring and Logging

Implement more comprehensive monitoring:

```python
# Add this to app.py

from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Start metrics server
start_http_server(8001)

# Define metrics
DOCUMENT_COUNTER = Counter('documents_processed_total', 'Total documents processed', ['status'])
PROCESSING_TIME = Histogram('document_processing_seconds', 'Time to process documents')
QUESTION_COUNTER = Counter('questions_answered_total', 'Total questions answered')
ACTIVE_JOBS = Gauge('active_jobs', 'Number of active processing jobs')

# Update process_document_task function
def process_document_task(job_id, file_path, target_language, generate_summary, extract_entities):
    try:
        # Update job status and metrics
        job_store[job_id]["status"] = "processing"
        ACTIVE_JOBS.inc()
        
        # Process document with timing
        start_time = time.time()
        result = process_document(file_path, target_language, generate_summary, extract_entities)
        processing_time = time.time() - start_time
        
        # Record metrics
        PROCESSING_TIME.observe(processing_time)
        DOCUMENT_COUNTER.labels(status="success").inc()
        
        # Update job with results
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = result
    except Exception as e:
        # Record failure metrics
        DOCUMENT_COUNTER.labels(status="failure").inc()
        
        # Update job with error
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        logger.error(f"Job {job_id} failed: {str(e)}")
    finally:
        # Always decrement active jobs
        ACTIVE_JOBS.dec()
```

## Deployment Considerations

### Docker Containerization

Create a Dockerfile for the application:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results static

# Expose ports
EXPOSE 8000
EXPOSE 8001

# Command to run the application
CMD ["python", "app.py"]
```

And a requirements.txt file:

```
fastapi==0.75.0
uvicorn==0.17.6
pydantic==1.9.0
python-multipart==0.0.5
torch==1.9.0
transformers==4.18.0
PyPDF2==2.10.5
python-docx==0.8.11
prometheus-client==0.14.1
```

### Scaling with Kubernetes

Create a Kubernetes deployment configuration:

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-processor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: document-processor
  template:
    metadata:
      labels:
        app: document-processor
    spec:
      containers:
      - name: document-processor
        image: document-processor:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        resources:
          limits:
            cpu: "2"
            memory: "8Gi"
            nvidia.com/gpu: 1
          requests:
            cpu: "1"
            memory: "4Gi"
        volumeMounts:
        - name: document-storage
          mountPath: /app/uploads
        - name: result-storage
          mountPath: /app/results
      volumes:
      - name: document-storage
        persistentVolumeClaim:
          claimName: document-storage-pvc
      - name: result-storage
        persistentVolumeClaim:
          claimName: result-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: document-processor-service
spec:
  selector:
    app: document-processor
  ports:
  - name: api
    port: 80
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001
  type: LoadBalancer
```

## Lessons Learned and Best Practices

In building this application, we've applied several best practices:

1. **Modular architecture**: We separated concerns into distinct components
2. **Error handling**: We implemented robust error handling and graceful degradation
3. **Asynchronous processing**: We used background tasks for long-running operations
4. **Monitoring and logging**: We added comprehensive logging and metrics
5. **Resource optimization**: We implemented lazy loading and batch processing
6. **Containerization**: We prepared the application for containerized deployment

Some key lessons from this project:

1. **Balance between features and performance**: More features can increase processing time
2. **Memory management is crucial**: Transformer models require careful memory management
3. **Error handling is essential**: NLP processing can fail in unexpected ways
4. **Batch processing improves efficiency**: Processing in batches is more efficient than one at a time
5. **Monitoring provides insights**: Comprehensive monitoring helps identify bottlenecks

## Resources for Continued Learning

To continue learning about Hugging Face transformers and related technologies, here are some valuable resources:

### Official Documentation

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/index)
- [Hugging Face Model Hub](https://huggingface.co/models)

### Books and Courses

- "Natural Language Processing with Transformers" by Lewis Tunstall, Leandro von Werra, and Thomas Wolf
- "Deep Learning for NLP and Speech Recognition" by Uday Kamath, John Liu, and James Whitaker
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)

### Research Papers

- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019)

### Community Resources

- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Hugging Face GitHub Repository](https://github.com/huggingface/transformers)
- [Papers with Code](https://paperswithcode.com/)

## Conclusion

In this final post of our "Mastering Hugging Face Transformers" series, we've built a complete multilingual document processing application that demonstrates many of the concepts covered throughout the series. We've shown how to:

- Process documents in multiple languages
- Extract key information using named entity recognition
- Generate summaries of document content
- Translate documents between languages
- Answer questions about document content
- Optimize for performance and resource usage
- Deploy the application in a production environment

This project brings together many aspects of transformer models, from basic understanding to advanced applications and deployment considerations. By following along, you've gained practical experience with the entire lifecycle of a transformer-based application.

As transformer models continue to evolve, the principles and patterns we've explored in this series will remain valuable. The modular architecture we've used allows for easy updates as new models and techniques emerge.

We hope this series has provided you with a solid foundation for working with Hugging Face transformers and that you'll continue to explore and build with these powerful tools!

---

*This concludes our "Mastering Hugging Face Transformers" series. Thank you for joining us on this journey through the world of transformer models!*