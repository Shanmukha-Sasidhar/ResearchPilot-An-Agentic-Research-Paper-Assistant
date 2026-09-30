# ResearchPilot — An Agentic Research Paper Assistant 📚🤖

ResearchPilot is an AI-powered research assistant that uses **Agentic RAG**, **hybrid retrieval (Dense + Sparse BM25)**, **cross-encoder reranking**, and **conversational memory** to help researchers, students, and engineers understand, explore, and interact with academic papers using natural language.

---

## 🌟 Key Features

- 📄 **Multimodal Document Parsing**: PDF parsing with PyMuPDF, `pdfplumber`, and image extraction.
- 🧩 **Semantic & Recursive Chunking**: Token-aware text chunking preserving page numbers and section hierarchy.
- 🔍 **Hybrid Retrieval Engine**:
  - **Dense Retrieval**: Semantic embeddings with Sentence-Transformers stored in ChromaDB / FAISS.
  - **Sparse Retrieval**: BM25 keyword matching for precise acronyms, equations, and technical terminology.
- 🎯 **Cross-Encoder Reranking**: Re-orders retrieved candidates for optimal semantic relevance before generation.
- 🤖 **Agentic Workflow (LangGraph)**: Multi-step reasoning agent with query routing, self-correction, and tool usage.
- 💬 **Interactive Streamlit UI**: User-friendly chat interface for uploading PDFs, asking questions, and viewing cited sources.
- 📊 **LangSmith Tracing**: Full observability and prompt tracing for agent decision steps.

---

## 📁 Repository Structure

```text
├── app/
│   ├── agent/            # LangGraph agent definitions, routing, and workflows
│   ├── ingestion/        # PDF parser, image extraction, and chunkers
│   ├── retrieval/        # Hybrid search (Dense Chroma/FAISS + BM25) and reranker
│   ├── generation/       # LLM response generation and prompt templates
│   ├── config.py         # Application configuration & paths
│   └── __init__.py
├── data/
│   ├── papers/           # Uploaded PDF research papers (.gitkeep)
│   └── processed/        # Vector stores and index caches (.gitkeep)
├── ui/
│   └── streamlit_app.py  # Streamlit conversational web interface
├── tests/                # Test suite
├── main.py               # Core orchestrator application
├── requirements.txt      # Project dependencies
├── .env.example          # Environment variables template
└── README.md             # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Shanmukha-Sasidhar/ResearchPilot-An-Agentic-Research-Paper-Assistant.git
cd ResearchPilot-An-Agentic-Research-Paper-Assistant
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory (based on `.env.example`):

```env
OPENROUTER_API_KEY="your-openrouter-api-key"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=research-paper-agent
```

### 5. Launch the Application

Run the Streamlit application:

```bash
streamlit run ui/streamlit_app.py
```

Or run via CLI:

```bash
python main.py
```

---

## 🛠️ Tech Stack

- **Frameworks & Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain)
- **Vector Storage**: [ChromaDB](https://www.trychroma.com/), [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings & Reranking**: [Sentence-Transformers](https://www.sbert.net/), [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- **Document Processing**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [pdfplumber](https://github.com/jsvine/pdfplumber)
- **User Interface**: [Streamlit](https://streamlit.io/)
- **Observability**: [LangSmith](https://smith.langchain.com/)

---

## 📄 License

This project is licensed under the MIT License.
