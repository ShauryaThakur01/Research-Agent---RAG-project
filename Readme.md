(Just a small note that if you want to view the code then please visit the pdf file since github had some problems with the original .ipynb file)

# Research Assistant Agent

This project implements a Research Assistant Agent designed to fetch academic papers, process their content, and provide answers to user queries using a Retrieval-Augmented Generation (RAG) pipeline. It also includes an evaluation framework using RAGAS to assess the quality of the RAG system.

## Project Overview

The agent performs the following steps:
1.  **Paper Fetching**: Retrieves relevant research papers from arXiv based on a specified query.
2.  **Document Processing**: Loads PDF documents, splits them into manageable chunks using a custom parent-document retrieval strategy.
3.  **Embedding & Storage**: Embeds the document chunks using HuggingFace embeddings and stores them in a Chroma vector database, along with an in-memory store for parent documents.
4.  **Retrieval-Augmented Generation (RAG)**: Constructs a RAG chain to answer user questions by retrieving relevant document chunks and passing them to a Large Language Model (LLM) from Groq.
5.  **Evaluation**: Evaluates the RAG system's performance using RAGAS metrics (Faithfulness, Answer Relevancy, Context Utilization).

## Stack

*   **LangChain**: For building the RAG pipeline components (document loading, text splitting, retrievers, chains, LLM integration).
*   **ChromaDB**: As the vector store for efficient similarity search of document embeddings.
*   **arXiv API**: For programmatically fetching research papers.
*   **RAGAS**: For evaluating the RAG system's performance on various metrics.
*   **Groq**: Provides fast LLM inference for both the RAG chain and RAGAS evaluation.
*   **HuggingFace Embeddings**: Used for generating document embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
*   **PyPDFLoader**: For loading PDF content from fetched papers.
*   **langchain-experimental**: For advanced retriever implementations (though a custom implementation is used here).

## Setup and Usage

### 1. Install Dependencies

Ensure you have all necessary libraries installed. The notebook includes `pip install` commands for these:

```python
!pip install -qU arxiv pypdf langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers langchain-groq datasets ragas litellm openai langchain_openai langchain_experimental
```

### 2. Set Up API Keys

This project requires an API key for Groq. You will be prompted to enter it when the relevant cells are run:

```python
import os
from getpass import getpass

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API Key: ")
```

### 3. Fetch and Process Documents

Papers are fetched from arXiv, loaded, and then processed using a custom parent-document chunking strategy. This involves splitting documents into smaller 'child' chunks for vector storage and larger 'parent' chunks for contextual retrieval.

### 4. Run the RAG Chain

A RAG chain is constructed to answer questions based on the retrieved documents. You can invoke it with your query:

```python
query = "What is RevengeBench? What is its purpose?"
response = rag_chain.invoke(query)
print(f"Query: {query}\n")
print(f"Answer:\n{response}")
```

### 5. Evaluate with RAGAS

The system's performance can be evaluated using RAGAS. An evaluation dataset is created, and metrics like Faithfulness, Answer Relevancy, and Context Utilization are calculated using Groq's LLMs as evaluators.

```python
# ... (Code for RAGAS setup and evaluation)

print("\nRunning RAGAS evaluation on Groq (Llama 3.3-70B is grading)....")
eval_result = evaluate(
    dataset=ragas_eval_dataset,
    metrics=metrics,
)

results_df = eval_result.to_pandas()
print("\n=== OVERALL RAGAS SCORES ===")
print(eval_result)
results_df[["user_input", "faithfulness", "answer_relevancy", "context_utilization"]]
```

## Chunking Strategies

The notebook explores two primary chunking strategies:

1.  **RecursiveCharacterTextSplitter**: A standard recursive splitting approach.
2.  **Custom Parent-Document Retrieval**: An advanced strategy where smaller chunks are used for vector search, but larger parent chunks are retrieved to provide more context to the LLM. This custom implementation was developed to work around `MultiVectorRetriever` import issues.
