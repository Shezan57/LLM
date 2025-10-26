# LLM (Shezan57)

Welcome — I reviewed the repository layout and contents to create a focused, detailed README that showcases your work and capabilities across LLMs, retrieval, embeddings, supervised learning, and building simple agents. The repository contains Jupyter notebooks, Python scripts, and helper modules covering practical experiments: embeddings, RAG with LangChain, fine-tuned agents, output parsing strategies, supervised sentiment tasks, topic modeling, multimodal notes, and tooling for monitoring and document extraction.

Note: The repository listing I inspected may be incomplete due to API result limits. You can view the repository in full here:
https://github.com/Shezan57/LLM/tree/main/

---

Table of contents
- Project overview
- Highlights — what this repo demonstrates about my skills
- Repository structure (file-by-file summary with links)
- Quickstart (local / Colab)
- Recommended environment & dependencies
- How to run selected demos
- Design choices & implementation notes (principles shown)
- Suggestions to improve / next steps
- Contributing & license

---

Project overview
This repo is a hands-on collection of experiments and demos that demonstrate practical knowledge of modern NLP/LLM workflows:
- Generating and prompting LLMs (text generation experiments)
- Building simple chat/agent systems and LLM-powered assistants (LangChain)
- Retrieval-Augmented Generation (RAG) examples using embeddings and vector indexes
- Training / using embedding models and applying them to semantic search
- Supervised classification (sentiment) experiments (including Rotten Tomatoes)
- Topic modeling and clustering
- Multimodal model notes and experiments
- Output parsing patterns (string / JSON / Pydantic)
- Small tools: document loaders, token usage monitoring, and utilities

This collection shows end-to-end competence from data processing, model usage (including LangChain & Hugging Face), to producing reproducible notebooks.

---

Highlights — what this repo demonstrates about my abilities
- Practical use of embeddings and semantic search for retrieval and RAG.
- Experience with LangChain: memory concepts, chains, and RAG construction.
- Knowledge of supervised model training and evaluation for sentiment classification.
- Understanding of output parsing strategies (structured outputs, pydantic validation).
- Integration of multiple LLM toolings (OpenAI / Google AI / transformer-based approaches).
- Notebook-driven exploratory data analysis and reproducible experiments.
- Ability to create utilities for document extraction, token monitoring, and small chat agents.

---

Repository structure (file-by-file)
Below I list the main files and notebooks present in the repository with a short description and suggestions for how each showcases your skill. Click the file name to view it on GitHub.

- Notebooks (interactive, exploratory experiments)
  - Agno_finance_agent.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/Agno_finance_agent.ipynb  
    Description: Notebook building/experimenting with a finance-focused agent. Likely demonstrates prompt design, custom tool integration, and finance-specific retrieval or heuristics.

  - Clusturing_and_topic_modeling.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/Clusturing_and_topic_modeling.ipynb  
    Description: Topic modeling and clustering experiments (unsupervised learning). Shows skill with preprocessing, vectorization (TF-IDF / embeddings), dimensionality reduction, and cluster inspection.

  - Multimodal_models.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/Multimodal_models.ipynb  
    Description: Notes/experiments about multimodal models (images + text). Demonstrates awareness of multimodal architectures and possible usage patterns.

  - My_First_AI_Agent.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/My_First_AI_Agent.ipynb  
    Description: A beginner-friendly agent notebook that likely shows building an interactive assistant using LLMs and tool connectors.

  - Supervised_classification_rotten_tomatoes.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/Supervised_classification_rotten_tomatoes.ipynb  
    Description: Supervised sentiment classification workflow on Rotten Tomatoes dataset. Demonstrates dataset preparation, feature engineering, training (classical ML or transformer-based), evaluation and visualization.

  - Training_embedding_model.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/Training_embedding_model.ipynb  
    Description: Experiments training or fine-tuning embeddings (could use sentence-transformers or Hugging Face). Shows knowledge of embedding loss functions and dataset construction.

  - rag_using_langchain.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/rag_using_langchain.ipynb  
    Description: A focused RAG pipeline using LangChain. Likely contains document ingestion, splitting, embedding, index creation (FAISS or similar), and a retrieval-augmented QA flow.

  - generating_first_text_LLM.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/generating_first_text_LLM.ipynb  
    Description: Introductory notebook showing text generation, prompts, sampling, temperature, and decoding strategies.

  - rotten_tomatoes_sentiment_classification.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/rotten_tomatoes_sentiment_classification.ipynb  
    Description: Another sentiment classification exploration — may complement the supervised notebook with deeper model experiments (possibly transformers).

  - text2text_transformer_sentiment_rotten_tomatoes.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/text2text_transformer_sentiment_rotten_tomatoes.ipynb  
    Description: Use of text-to-text transformer (T5-like) for sentiment mapping. Shows knowledge of sequence-to-sequence training for classification/regression tasks.

  - song_embedding_model.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/song_embedding_model.ipynb  
    Description: Building embeddings for music/song metadata or lyrics for similarity/search tasks.

  - using_langchain.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/using_langchain.ipynb  
    Description: LangChain primer — shows how to wire LLMs, chains, and simple prompt chains.

  - using_langchain_memory_concept.ipynb  
    URL: https://github.com/Shezan57/LLM/blob/main/using_langchain_memory_concept.ipynb  
    Description: Experiments with LangChain memory modules (conversation history, vector memory) — demonstrates design choices for persistent context and handling state.

- Python scripts & helpers
  - document_loader.py  
    URL: https://github.com/Shezan57/LLM/blob/main/document_loader.py  
    Description: (Empty file in current listing) Intended to contain custom document loading logic for ingestion pipelines.

  - extracting-info-from-doc.py  
    URL: https://github.com/Shezan57/LLM/blob/main/extracting-info-from-doc.py  
    Description: Script to extract structured information from documents (likely uses regex, OCR, or LLM extraction).

  - google_ai.py  
    URL: https://github.com/Shezan57/LLM/blob/main/google_ai.py  
    Description: Small helper integrating Google AI APIs or example usage.

  - information-retrieval-with-tools.py  
    URL: https://github.com/Shezan57/LLM/blob/main/information-retrieval-with-tools.py  
    Description: Demonstrates retrieval pipelines combined with external tools (search, file loaders).

  - json-output-parser.py  
    URL: https://github.com/Shezan57/LLM/blob/main/json-output-parser.py  
    Description: Example showing how to parse LLM outputs into JSON safely (parsing + validation).

  - medical_bot_deepseek_fine_tuned.py  
    URL: https://github.com/Shezan57/LLM/blob/main/medical_bot_deepseek_fine_tuned.py  
    Description: Medical assistant that uses a fine-tuned model or integration with Deepseek. Demonstrates domain adaptation and safety considerations.

  - monitoring-token-usage.py  
    URL: https://github.com/Shezan57/LLM/blob/main/monitoring-token-usage.py  
    Description: Utility to track and report LLM token usage — useful for cost monitoring.

  - pydantic-output-parser.py  
    URL: https://github.com/Shezan57/LLM/blob/main/pydantic-output-parser.py  
    Description: Example of using Pydantic models to validate and structure LLM outputs.

  - pydantic_prac.py  
    URL: https://github.com/Shezan57/LLM/blob/main/pydantic_prac.py  
    Description: Practice script demonstrating Pydantic usage and model validation.

  - question_answering.py  
    URL: https://github.com/Shezan57/LLM/blob/main/question_answering.py  
    Description: Small QA utility (RAG or direct LLM QA) showing pipeline for answering user queries from documents.

  - simple_chat_bot.py  
    URL: https://github.com/Shezan57/LLM/blob/main/simple_chat_bot.py  
    Description: Lightweight chat bot example that demonstrates sessions, prompt history, and LLM calls.

  - str-output-parser.py  
    URL: https://github.com/Shezan57/LLM/blob/main/str-output-parser.py  
    Description: Parse raw string outputs into structured forms (examples for regex or simple delimiters).

  - structured-output-parser.py  
    URL: https://github.com/Shezan57/LLM/blob/main/structured-output-parser.py  
    Description: Another strategy for structured parsing; showcases different parsing patterns.

  - structured_output_pydantic.py  
    URL: https://github.com/Shezan57/LLM/blob/main/structured_output_pydantic.py  
    Description: Combining structured output techniques with Pydantic for robust validation.

- Directories
  - .devcontainer/  
    URL: https://github.com/Shezan57/LLM/tree/main/.devcontainer  
    Description: Dev container config (VS Code devcontainer) — helpful for reproducible environment (if populated).
  - .idea/  
    URL: https://github.com/Shezan57/LLM/tree/main/.idea  
    Description: IDE settings (can be ignored or removed from repo).
  - data/  
    URL: https://github.com/Shezan57/LLM/tree/main/data  
    Description: Data directory (empty in listing). Good place for small demo datasets or pointers to download scripts.
  - document_loader/  
    URL: https://github.com/Shezan57/LLM/tree/main/document_loader  
    Description: Directory for document ingestion utilities (empty in listing).
  - text_splitter/  
    URL: https://github.com/Shezan57/LLM/tree/main/text_splitter  
    Description: Text splitting utilities for RAG pipelines (empty in listing).

---

Quickstart (local / Colab)
1. Clone the repo:
   git clone https://github.com/Shezan57/LLM.git
2. Create a virtual environment and install Python dependencies (example):
   python -m venv .venv
   source .venv/bin/activate          # Unix / Mac
   .venv\Scripts\activate             # Windows
3. Install (recommended set):
   pip install -U pip
   pip install jupyterlab notebook pandas numpy scikit-learn matplotlib seaborn sentence-transformers transformers torch faiss-cpu langchain openai pydantic huggingface-hub
   (Adjust CPU/GPU packages for your machine; if using GPU, install torch with CUDA support following official instructions.)
4. Start JupyterLab/Notebook and open notebooks:
   jupyter lab
   or
   jupyter notebook

Notes for Colab: Upload the repository or use `git clone` inside Colab, then run pip installs in a cell. For heavy model training, prefer GPU runtimes.

---

Recommended environment & dependencies
- Python >= 3.8
- JupyterLab / Notebook
- Core libraries: numpy, pandas, scikit-learn, matplotlib, seaborn
- LLM / embeddings: transformers, sentence-transformers, torch, huggingface-hub
- Retrieval & agents: langchain, faiss-cpu (or faiss-gpu), openai, pydantic
- Optional: openai/other provider SDKs, google-cloud-aiplatform (if using Google AI), streamlit (for simple demos)

Create a requirements.txt (example):
numpy
pandas
scikit-learn
matplotlib
seaborn
jupyterlab
torch
transformers
sentence-transformers
faiss-cpu
langchain
openai
pydantic
huggingface-hub

---

How to run selected demos (suggestions & examples)

1) RAG with LangChain (rag_using_langchain.ipynb)
- Purpose: ingest documents, split text, compute embeddings, index into FAISS, run retrieval + generation.
- Steps:
  - Install dependencies (sentence-transformers, faiss-cpu).
  - Provide OPENAI_API_KEY (or use HuggingFace models for generation & embeddings).
  - Run notebook cells sequentially to ingest, embed, build index, and query.

2) Supervised sentiment classification (Supervised_classification_rotten_tomatoes.ipynb)
- Purpose: Data preprocessing, vectorization, classifier training, evaluation.
- Steps:
  - Download Rotten Tomatoes dataset (or use dataset loading utilities).
  - Run preprocessing cells, fit model, inspect accuracy/confusion matrix.

3) Embedding training (Training_embedding_model.ipynb)
- Purpose: Fine-tune or train embedding model for semantic tasks.
- Steps:
  - Prepare paired examples or triplet data.
  - Use sentence-transformers or HuggingFace trainer patterns.
  - Evaluate via retrieval / similarity tasks.

4) Output parsing strategies (pydantic-output-parser.py / json-output-parser.py)
- Purpose: Show robust ways to parse LLM responses into structured data.
- Recommendation: Use Pydantic models to validate results and handle edge cases.

5) Simple chatbot (simple_chat_bot.py)
- Purpose: Minimal REPL or small web demo to interact with an LLM; shows session handling and prompt formatting.

6) Token monitoring (monitoring-token-usage.py)
- Purpose: Track usage from OpenAI responses; helpful for production cost monitoring.

---

Design choices & implementation notes (principles shown)
- Notebook-first experimentation: notebooks present exploratory, reproducible workflows.
- Modularity: scripts and helper modules isolate utilities (parsers, loaders, token monitors).
- Robust parsing: Using Pydantic and multiple parsing strategies prevents brittle downstream logic.
- Reproducibility: Devcontainer and environment recommendations for reproducible setups (if .devcontainer populated).
- Safety & domain adaptation: Examples like medical_bot_deepseek_fine_tuned.py demonstrate domain-specific tuning and caution that medical advice requires careful validation.

---

Suggestions to strengthen this repository
- Add a top-level requirements.txt and a short Makefile or scripts to set up the environment quickly.
- Add small README snippets into key directories (document_loader/, text_splitter/) to explain purpose and expected files.
- Populate document_loader/ and text_splitter/ with ready-to-run utilities and example files.
- Add a small runnable demo (Streamlit or Flask) that wires one notebook's pipeline into a simple web UI for an interactive portfolio piece.
- Add unit tests for core parsers (json-output-parser, pydantic-output-parser) to demonstrate engineering rigor.
- Add LICENSE and CODE_OF_CONDUCT if you want public contributions.

---

Next steps (what I did and how I can help further)
I inspected the repository structure and created this README to highlight your strengths and give runnable guidance for each component. If you'd like, I can:
- Generate a concrete requirements.txt and devcontainer config (populate .devcontainer) so anyone can reproduce your environment.
- Open the key Python files and notebooks and produce more precise README descriptions and examples (I can parse the notebooks and extract the key code/cell outputs).
- Create a minimal Streamlit demo that wires rag_using_langchain.ipynb into an interactive app.

Tell me which improvement you prefer and I will implement it (e.g., create requirements.txt, populate README further with code snippets from specific notebooks, or scaffold a demo app). I can also push a README.md file directly to the repository if you want — say "create README" and I will add it.

Thank you — this repo contains strong, practical experiments across the LLM stack and with a few polish items it will make an excellent portfolio that showcases both research and applied engineering skills.
