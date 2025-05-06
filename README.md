# Repository Overview

This repository serves as a comprehensive collection of resources and practical examples in the field of Artificial Intelligence, with a focus on Deep Learning, Natural Language Processing (NLP), and Large Language Models (LLMs).

## Contents

### 1. PyTorch Fundamentals & Workflow (`pytorch_all.ipynb`)
- **Tensor Operations**: Demonstrations of essential PyTorch tensor manipulations including creation, reshaping, stacking, squeezing, permuting, and indexing.
- **NumPy Integration**: Examples of converting between PyTorch tensors and NumPy arrays.
- **GPU Acceleration**: How to check for and utilize CUDA-enabled GPUs for tensor operations.
- **Reproducibility**: Using `random.seed()` for consistent results.
- **End-to-End PyTorch Workflow**:
    - Data preparation and loading (e.g., for linear regression).
    - Splitting data into training and testing sets.
    - Building custom models using `torch.nn.Module`, `nn.Parameter`.
    - Defining `forward` pass.
    - Making predictions with `torch.inference_mode()`.
    - Plotting results and model performance.

### 2. Hugging Face Transformers (`Hugging_face_star.ipynb`)
- **Introduction to `pipeline`**: Utilizing the high-level `pipeline` function for various NLP tasks.
    - `sentiment-analysis`
    - `zero-shot-classification`
    - `text-generation` (e.g., using `distilgpt2`)
    - `fill-mask`
    - `ner` (Named Entity Recognition)
    - `question-answering`
    - `translation`
- **Transformer Architecture**:
    - Conceptual overview of Encoder, Decoder, and Encoder-Decoder models.
    - Understanding the role of self-attention.
    - Use cases for different architectures (NLU vs. NLG).
- **Inside the `pipeline`**: A breakdown of the tokenization, model processing, and post-processing steps for both PyTorch and TensorFlow backends.
- **Model Instantiation**: Using `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, and `AutoConfig` to load and configure pretrained models.
- **Transfer Learning**: The significance and application of transfer learning in NLP.
- **Environmental Considerations**: A note on the carbon footprint of training large models.

### 3. Hands-on Large Language Models (LLMs) (`Hands_on_LLM.ipynb`)
- **Text Classification**:
    - **Representation Models**: Sentiment analysis of movie reviews (Rotten Tomatoes dataset) using `transformers` pipelines (e.g., `cardiffnlp/twitter-roberta-base-sentiment-latest`).
    - **Embeddings**: Using sentence transformers (e.g., `sentence-transformers/all-mpnet-base-v2`) for generating embeddings and performing classification.
- **Generative Models & Langchain**:
    - **Loading Quantized Models**: Implementing `LlamaCpp` with models like `microsoft/Phi-3-mini-4k-instruct-gguf`.
    - **Chat Models**: Interacting with `ChatGoogleGenerativeAI` (Gemini).
    - **Agent Frameworks**: Building ReAct agents with tools like DuckDuckGo search, leveraging Langchain for complex reasoning and action tasks.
- **Text Clustering and Topic Modeling**:
    - Applying unsupervised learning techniques to group documents (e.g., Arxiv NLP dataset abstracts).
    - Common pipeline: embeddings -> dimensionality reduction -> clustering.

### 4. Interactive Notebooks
- **Colab Exploration (`colab1.ipynb`)**: An empty notebook, ready for interactive experiments and explorations in Google Colab.
- The repository includes Jupyter notebooks designed for hands-on learning and experimentation with the concepts mentioned above.

## Getting Started

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the desired notebook.
3.  Ensure you have the necessary dependencies installed. Common libraries include `torch`, `transformers`, `datasets`, `sklearn`, `sentence-transformers`, `langchain`, `llama-cpp-python`, `google-generativeai`. You might need to install them via pip:
    ```bash
    pip install torch transformers datasets scikit-learn sentence-transformers langchain llama-cpp-python google-generativeai matplotlib
    ```
4.  Open the notebooks using Jupyter Lab/Notebook or Google Colab.

Explore the contents to learn and experiment with modern machine learning and deep learning techniques.
