# Neural Text Summarization using Seq2Seq and Bahdanau Attention

## Overview

This project implements an abstractive text summarization model from scratch using PyTorch.

The primary goal of this project was to understand and implement the core concepts behind sequence-to-sequence learning, attention mechanisms, and neural text generation rather than relying on pre-trained transformer models.

The model was trained on the XSum dataset and deployed through a simple web application interface.

---

## Features

* Text preprocessing and vocabulary construction
* Encoder-Decoder architecture using LSTMs
* Bahdanau Attention mechanism
* Teacher forcing during training
* Greedy decoding during inference
* Training and validation pipeline
* Web interface for generating summaries
* Model deployment ready for Render

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

Dataset Used:

XSum (Extreme Summarization Dataset)

The dataset contains BBC news articles paired with single-sentence summaries.

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Datasets
* Flask
* HTML/CSS

---

## Learning Outcomes

Through this project, I implemented:

* Sequence-to-Sequence models from scratch
* Attention mechanisms
* Teacher forcing
* Neural text generation
* Dataset preprocessing pipelines
* Training and evaluation workflows
* Model deployment fundamentals

---

## Results

The project successfully demonstrates the complete workflow of neural text summarization.

Since the model is a relatively small LSTM-based architecture trained from scratch, generated summaries are not expected to match the performance of modern transformer-based models such as BART, T5, or PEGASUS.

The focus of this project is understanding the underlying NLP concepts and implementing them end-to-end.

---

## Future Improvements

* Beam Search Decoding
* Larger Vocabulary
* Pretrained Word Embeddings
* Transformer-based Architectures
* ROUGE Evaluation Metrics
* Improved Inference Pipeline

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
