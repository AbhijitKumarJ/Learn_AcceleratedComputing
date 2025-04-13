# Specialized NLP Tasks

Welcome to the fifth installment of our "Mastering Hugging Face Transformers" series! In our previous posts, we introduced transformer models, explored their architecture, learned how to use pre-trained models, and covered fine-tuning techniques. Now, we'll dive into specialized NLP tasks that you can tackle with Hugging Face transformers.

In this post, we'll explore five key NLP tasks in depth: text classification, named entity recognition, question answering, text summarization, and machine translation. For each task, we'll discuss the appropriate models, data preparation techniques, and implementation details.

## Text Classification and Sentiment Analysis

### Understanding Text Classification

Text classification involves assigning predefined categories to text documents. Common applications include:

- **Sentiment analysis**: Determining if text expresses positive, negative, or neutral sentiment
- **Topic classification**: Categorizing text by subject matter
- **Intent detection**: Identifying user intentions in conversational AI
- **Spam detection**: Filtering unwanted messages

### Models for Text Classification

Many transformer models excel at classification tasks. Some popular choices include:

- **BERT and RoBERTa**: Strong general-purpose encoders
- **DistilBERT**: A lighter, faster version of BERT
- **XLNet**: Particularly good for longer documents
- **ALBERT**: A memory-efficient version of BERT

### Implementing Sentiment Analysis

Let's implement a sentiment analysis solution using Hugging Face:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_dataset

# Load a sentiment analysis dataset
dataset = load_dataset("imdb")
print(f"Dataset size: {len(dataset['train'])} training, {len(dataset['test'])} test examples")

# Examine an example
print(dataset["train"][0])

# Prepare the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare for training
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Load pre-trained model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# Training setup (similar to previous post)
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']}")

# Create a pipeline for inference
sentiment_analyzer = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer
)

# Test on new examples
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

### Multi-label Classification

Some classification tasks require assigning multiple labels to a single text. For example, a news article might belong to both "Politics" and "Economy" categories:

```python
# Multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,  # Number of possible labels
    problem_type="multi_label_classification"
)

# For multi-label, use binary cross-entropy loss
# and sigmoid activation (instead of softmax)
```

### Zero-shot Classification

Zero-shot classification allows classifying text into categories not seen during training:

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "This restaurant has amazing pizza and great service."
candidate_labels = ["food", "service", "ambiance", "price"]

result = classifier(text, candidate_labels)
print(f"Text: {text}")
for label, score in zip(result["labels"], result["scores"]):
    print(f"{label}: {score:.4f}")
```

## Named Entity Recognition

### Understanding Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies named entities in text into predefined categories such as:

- Person names
- Organizations
- Locations
- Date and time expressions
- Monetary values
- Product names

NER is crucial for information extraction, question answering, and knowledge graph construction.

### Models for NER

Transformer models for NER are typically fine-tuned for token classification:

- **BERT/RoBERTa with token classification heads**
- **DistilBERT-based token classifiers**
- **Specialized models like Flair**

### Implementing NER

Let's implement a named entity recognition solution:

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from datasets import load_dataset

# Load a NER dataset (CoNLL-2003)
dataset = load_dataset("conll2003")

# Examine the dataset
print(dataset)
print(dataset["train"][0])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens have word_id = None
            if word_idx is None:
                label_ids.append(-100)
            # For the first token of a word, use the corresponding label
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent tokens of a word, use the same label
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize and align labels
tokenized_datasets = dataset.map(
    tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names
)

# Load pre-trained model with token classification head
id2label = {
    0: "O",       # Outside of a named entity
    1: "B-PER",   # Beginning of person name
    2: "I-PER",   # Inside of person name
    3: "B-ORG",   # Beginning of organization
    4: "I-ORG",   # Inside of organization
    5: "B-LOC",   # Beginning of location
    6: "I-LOC",   # Inside of location
    7: "B-MISC",  # Beginning of miscellaneous
    8: "I-MISC"   # Inside of miscellaneous
}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=9,
    id2label=id2label,
    label2id=label2id
)

# Training setup
from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np

metric = load_metric("seqeval")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Create a pipeline for inference
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Test on new examples
text = "Apple Inc. is planning to open a new office in New York City next year, according to CEO Tim Cook."
results = ner(text)

for entity in results:
    print(f"Entity: {entity['word']}")
    print(f"Type: {entity['entity_group']}")
    print(f"Score: {entity['score']:.4f}")
    print(f"Start: {entity['start']}, End: {entity['end']}")
    print("---")
```

### Custom NER Models

For domain-specific NER (e.g., medical or legal entities), you'll need to fine-tune on domain data:

```python
# Example: Creating a custom NER dataset
from datasets import Dataset

# Your annotated data
texts = ["Patient John Smith was diagnosed with hypertension."]
tokens = [["Patient", "John", "Smith", "was", "diagnosed", "with", "hypertension", "."]]
ner_tags = [[0, 1, 2, 0, 0, 0, 3, 0]]  # 0=O, 1=B-PERSON, 2=I-PERSON, 3=B-CONDITION

# Create dataset
custom_dataset = Dataset.from_dict({"tokens": tokens, "ner_tags": ner_tags})

# Then proceed with tokenization and training as above
```

## Question Answering

### Understanding Question Answering

Question answering (QA) systems answer natural language questions by extracting answers from a given context or generating them from learned knowledge. Types of QA include:

- **Extractive QA**: Finding the answer span within a provided context
- **Abstractive QA**: Generating an answer that might not appear verbatim in the context
- **Open-domain QA**: Answering questions without a specific context provided

### Models for Question Answering

Popular transformer models for QA include:

- **BERT/RoBERTa with QA heads**: For extractive QA
- **T5/BART**: For abstractive QA
- **DPR (Dense Passage Retrieval)**: For open-domain QA

### Implementing Extractive QA

Let's implement an extractive question answering system:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from datasets import load_dataset

# Load a QA dataset (SQuAD)
dataset = load_dataset("squad")

# Examine the dataset
print(dataset)
print(dataset["train"][0])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        
        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise, find the token positions that contain the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Tokenize and prepare the dataset
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Load pre-trained model with QA head
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# Training setup
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()

# Create a pipeline for inference
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Test on new examples
context = """
Hugging Face is an AI company that develops tools for building applications using machine learning. 
It was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf. 
The company is based in New York City and Paris, France. 
Hugging Face is most known for its Transformers library, which provides pre-trained models for natural language processing tasks.
"""

questions = [
    "When was Hugging Face founded?",
    "Who founded Hugging Face?",
    "Where is Hugging Face based?",
    "What is Hugging Face known for?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Score: {result['score']:.4f}")
    print("---")
```

### Open-Domain Question Answering

Open-domain QA combines retrieval and reading comprehension:

```python
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRReader
from transformers import AutoTokenizer, pipeline

# Load DPR components
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

# Example knowledge base (in practice, this would be much larger)
passages = [
    "Hugging Face was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf.",
    "Hugging Face is based in New York City and Paris, France.",
    "The Transformers library provides pre-trained models for NLP tasks.",
    "Hugging Face Transformers supports PyTorch, TensorFlow, and JAX."
]

# Encode passages (would be done offline for a real system)
import torch

passage_embeddings = []
for passage in passages:
    inputs = context_tokenizer(passage, return_tensors="pt")
    with torch.no_grad():
        embedding = context_encoder(**inputs).pooler_output
    passage_embeddings.append(embedding)

# Function to retrieve relevant passages
def retrieve(question, top_k=2):
    # Encode the question
    inputs = question_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        question_embedding = question_encoder(**inputs).pooler_output
    
    # Calculate similarity scores
    scores = []
    for passage_embedding in passage_embeddings:
        score = torch.matmul(question_embedding, passage_embedding.t()).item()
        scores.append(score)
    
    # Get top-k passages
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [passages[i] for i in top_indices]

# Function to read the answer from retrieved passages
def answer_question(question):
    # Retrieve relevant passages
    retrieved_passages = retrieve(question)
    
    # Prepare inputs for the reader
    inputs = reader_tokenizer(
        questions=[question] * len(retrieved_passages),
        titles=["Title"] * len(retrieved_passages),
        texts=retrieved_passages,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # Get answer spans
    with torch.no_grad():
        outputs = reader(**inputs)
    
    # Process outputs
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    relevance_logits = outputs.relevance_logits
    
    # Get the most relevant passage
    relevance_scores = torch.nn.functional.softmax(relevance_logits, dim=0)
    best_passage_idx = torch.argmax(relevance_scores).item()
    
    # Get the best answer span from the most relevant passage
    start_idx = torch.argmax(start_logits[best_passage_idx]).item()
    end_idx = torch.argmax(end_logits[best_passage_idx]).item()
    
    # Decode the answer
    input_ids = inputs.input_ids[best_passage_idx]
    answer = reader_tokenizer.decode(input_ids[start_idx:end_idx+1])
    
    return {
        "answer": answer,
        "passage": retrieved_passages[best_passage_idx],
        "relevance": relevance_scores[best_passage_idx].item()
    }

# Test the open-domain QA system
questions = [
    "Who founded Hugging Face?",
    "Where is Hugging Face based?",
    "What frameworks does Transformers support?"
]

for question in questions:
    result = answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"From passage: {result['passage']}")
    print(f"Relevance score: {result['relevance']:.4f}")
    print("---")
```

## Text Summarization

### Understanding Text Summarization

Text summarization condenses a longer text while preserving its key information. There are two main approaches:

- **Extractive summarization**: Selecting and combining existing sentences from the source text
- **Abstractive summarization**: Generating new sentences that capture the essence of the source text

Transformer models excel at abstractive summarization, which requires understanding the content and generating coherent summaries.

### Models for Summarization

Popular transformer models for summarization include:

- **BART**: Particularly strong for abstractive summarization
- **T5**: Versatile sequence-to-sequence model that performs well on summarization
- **Pegasus**: Specifically pre-trained for summarization tasks
- **LED (Longformer Encoder-Decoder)**: Handles longer documents efficiently

### Implementing Abstractive Summarization

Let's implement an abstractive summarization solution:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset

# Load a summarization dataset (CNN/DailyMail)
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Examine the dataset
print(dataset)
print(dataset["train"][0]["article"][:500])  # Show beginning of an article
print("\nSummary:")
print(dataset["train"][0]["highlights"])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(
        inputs, max_length=1024, truncation=True, padding="max_length"
    )
    
    # Setup the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["highlights"], max_length=128, truncation=True, padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process the dataset
tokenized_datasets = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)

# Load pre-trained summarization model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Training setup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
import numpy as np

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.split("\n")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split("\n")) for label in decoded_labels]
    
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    learning_rate=3e-5,
    num_train_epochs=1,  # For demonstration; use more epochs in practice
    evaluation_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # Subset for demonstration
    eval_dataset=tokenized_datasets["validation"].select(range(100)),
    compute_metrics=compute_metrics,
)

# Train the model (commented out as it's resource-intensive)
# trainer.train()

# Create a pipeline for inference
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Test on new examples
article = """
Hugging Face has emerged as a leading platform in the AI community, particularly for natural language processing (NLP) tasks. 
The company's Transformers library provides easy access to state-of-the-art pre-trained models, significantly lowering the barrier to entry for developers and researchers. 
Beyond just providing tools, Hugging Face has fostered a collaborative ecosystem where users can share and discover models, datasets, and applications. 
This approach has accelerated innovation in the field, as practitioners can build upon each other's work rather than starting from scratch. 
The Model Hub now hosts thousands of models covering various languages and specialized domains, from biomedical text analysis to legal document processing. 
Recently, Hugging Face has expanded beyond NLP into computer vision, audio processing, and reinforcement learning, positioning itself as a comprehensive platform for machine learning development.
"""

summary = summarizer(article, max_length=100, min_length=30, do_sample=False)
print("Original article length:", len(article.split()))
print("Summary length:", len(summary[0]["summary_text"].split()))
print("\nSummary:")
print(summary[0]["summary_text"])
```

### Controlling Summary Length and Style

You can control various aspects of the generated summaries:

```python
# Control summary length
short_summary = summarizer(article, max_length=50, min_length=10)
medium_summary = summarizer(article, max_length=100, min_length=30)
long_summary = summarizer(article, max_length=150, min_length=50)

# Control generation strategy
deterministic_summary = summarizer(article, do_sample=False)  # Greedy decoding
diverse_summary = summarizer(
    article, 
    do_sample=True,  # Sampling-based generation
    top_k=50,        # Consider top 50 tokens
    top_p=0.95,      # Nucleus sampling
    temperature=0.7  # Lower = more focused, higher = more diverse
)

# Generate multiple summaries
num_return_sequences = 3
multiple_summaries = summarizer(
    article,
    num_beams=4,
    num_return_sequences=num_return_sequences,
    do_sample=True
)

for i, summary in enumerate(multiple_summaries):
    print(f"Summary {i+1}:")
    print(summary["summary_text"])
    print("---")
```

## Machine Translation

### Understanding Machine Translation

Machine translation converts text from one language to another while preserving meaning. Neural machine translation (NMT) using transformer models has significantly improved translation quality in recent years.

Translation systems need to handle:

- **Linguistic differences**: Grammar, word order, idioms
- **Cultural nuances**: Context-dependent meanings
- **Domain-specific terminology**: Technical, legal, medical vocabulary

### Models for Translation

Popular transformer models for translation include:

- **mBART**: Multilingual BART for many language pairs
- **T5/mT5**: Text-to-text models supporting translation
- **OPUS-MT**: Specialized translation models for specific language pairs
- **M2M100**: Many-to-many multilingual translation model

### Implementing Machine Translation

Let's implement a machine translation solution:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset

# Load a translation dataset (WMT)
dataset = load_dataset("wmt16", "de-en")

# Examine the dataset
print(dataset)
print(dataset["train"][0]["translation"])

# Prepare the dataset
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

def preprocess_function(examples):
    inputs = [ex["de"] for ex in examples["translation"]]
    targets = [ex["en"] for ex in examples["translation"]]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Setup the targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Process the dataset
tokenized_datasets = dataset.map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names
)

# Load pre-trained translation model
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

# Training setup
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_metric
import numpy as np

metric = load_metric("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # SacreBLEU expects a list of references for each prediction
    decoded_labels = [[label] for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    learning_rate=3e-5,
    num_train_epochs=1,  # For demonstration; use more epochs in practice
    evaluation_strategy="epoch",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].select(range(1000)),  # Subset for demonstration
    eval_dataset=tokenized_datasets["validation"].select(range(100)),
    compute_metrics=compute_metrics,
)

# Train the model (commented out as it's resource-intensive)
# trainer.train()

# Create a pipeline for inference
translator = pipeline("translation_de_to_en", model="Helsinki-NLP/opus-mt-de-en")

# Test on new examples
german_texts = [
    "Maschinelles Lernen ist ein Teilgebiet der künstlichen Intelligenz.",
    "Die Transformer-Architektur hat die natürliche Sprachverarbeitung revolutioniert.",
    "Hugging Face bietet eine Plattform für den Austausch von vortrainierten Modellen."
]

for text in german_texts:
    translation = translator(text, max_length=100)
    print(f"German: {text}")
    print(f"English: {translation[0]['translation_text']}")
    print("---")
```

### Multilingual Translation

For translating between multiple language pairs, you can use multilingual models:

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load multilingual model
model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Function for translation between any supported language pair
def translate(text, source_lang, target_lang):
    # Set source language
    tokenizer.src_lang = source_lang
    
    # Tokenize
    encoded = tokenizer(text, return_tensors="pt")
    
    # Generate translation
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
    )
    
    # Decode
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Test with different language pairs
texts = {
    "en": "Machine learning is transforming the world.",
    "fr": "L'intelligence artificielle est l'avenir de la technologie.",
    "es": "Los modelos de lenguaje son muy útiles para muchas tareas.",
    "de": "Neuronale Netze können komplexe Probleme lösen."
}

# Translate each text to all other languages
for source_lang, text in texts.items():
    print(f"Original ({source_lang}): {text}")
    for target_lang in texts.keys():
        if target_lang != source_lang:
            translation = translate(text, source_lang, target_lang)
            print(f"  → {target_lang}: {translation}")
    print("---")
```

## Looking Ahead

In this post, we've explored five key NLP tasks and how to implement them using Hugging Face Transformers:

- Text classification and sentiment analysis for categorizing content
- Named entity recognition for extracting structured information
- Question answering for building interactive information systems
- Text summarization for condensing long documents
- Machine translation for breaking language barriers

These specialized tasks form the foundation of many real-world NLP applications. By understanding how to implement them with transformer models, you're well-equipped to build sophisticated language processing systems.

In our next post, we'll expand beyond text to explore multimodal applications, including vision transformers, audio transformers, and cross-modal models that can process multiple types of data simultaneously.

Stay tuned for "Multimodal Applications," where we'll discover how transformers are revolutionizing computer vision, speech processing, and more!

---

*This blog post is part of our "Mastering Hugging Face Transformers" series. Check back for new installments covering advanced topics, practical applications, and best practices.*