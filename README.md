# Neural Text Summarization using Seq2Seq and Bahdanau Attention

## Overview

This project implements an abstractive text summarization model from scratch using PyTorch.

The primary goal of this project was to understand and implement the core concepts behind sequence-to-sequence learning, attention mechanisms, and neural text generation rather than relying on pre-trained transformer models.

The model was trained on the XSum dataset and deployed through a web application built with Flask.

---

## Live Demo

🌐 **Try the deployed application here:**

[**https://nlp-text-summarizer-m587.onrender.com/**](https://nlp-text-summarizer-m587.onrender.com/)

> Note: The application may take a few seconds to start if the Render free tier instance is inactive.

---

## Features

* Text preprocessing and vocabulary construction
* Encoder-Decoder architecture using LSTMs
* Bahdanau Attention mechanism
* Teacher forcing during training
* Greedy decoding during inference
* Training and validation pipeline
* Flask-based web interface for generating summaries
* Deployment on Render

---

## Architecture

### Encoder

* Embedding Layer
* LSTM Layer
* Generates encoder hidden states and final context

### Attention

* Bahdanau Additive Attention
* Computes attention weights over encoder outputs
* Produces context vectors for the decoder

### Decoder

* Embedding Layer
* Attention Context
* LSTM Layer
* Linear Projection to Vocabulary Space

---

## Dataset

**Dataset Used:** XSum (Extreme Summarization Dataset)

The dataset contains BBC news articles paired with single-sentence summaries, making it a challenging benchmark for abstractive summarization.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Datasets
* Flask
* HTML/CSS
* Render

---

## Learning Outcomes

Through this project, I implemented:

* Sequence-to-Sequence models from scratch
* Encoder-Decoder architectures
* Bahdanau Attention mechanisms
* Teacher forcing during training
* Neural text generation
* Dataset preprocessing pipelines
* Training and validation workflows
* Model serialization and loading
* Web deployment fundamentals

---

## Results

The project successfully demonstrates the complete workflow of neural text summarization, from data preprocessing and model training to deployment.

Since the model is a relatively small LSTM-based architecture trained from scratch, generated summaries are not expected to match the performance of modern transformer-based models such as BART, T5, or PEGASUS.

The primary objective of this project was to gain a deeper understanding of the underlying NLP concepts and implement them end-to-end without relying on pre-trained summarization models.

---

## Future Improvements

* Beam Search Decoding
* Larger Vocabulary
* Pretrained Word Embeddings
* Transformer-based Architectures
* ROUGE Evaluation Metrics
* Improved Inference Pipeline
* Better Handling of Unknown Tokens
* Enhanced Summary Quality through Advanced Decoding Strategies

---

## Project Structure

```text
app.py
train.py
summarizer.py
model.pth
vocab.pkl
templates/
static/
notebooks/basic_nlp.ipynb
```

---

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/THANUJ-IMAYAVARAMBAN31/nlp.git
cd nlp
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser

Visit:

http://127.0.0.1:5000

---

## Deployment

The application is deployed on Render and can be accessed at:

[**https://nlp-text-summarizer-m587.onrender.com/**](https://nlp-text-summarizer-m587.onrender.com/)

This deployment loads the trained model and vocabulary files to perform inference through a simple web interface, allowing users to generate summaries directly from their browser.
