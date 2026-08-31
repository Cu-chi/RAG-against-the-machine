*This project has been created as part of the 42 curriculum by equentin.*

# RAG against the machine

# Description
This project implements a Retrieval-Augmented Generation (RAG) pipeline from scratch to answer questions based on a specific codebase (vLLM). The goal is to ingest and index documentation and Python code, retrieve the most relevant snippets for a given user query using a lexical search engine, and ultimately generate a grounded, hallucination-free answer using a small local Large Language Model (Qwen3-0.6B).

The project uses vLLM but we can produce same result with any codebase or any other types of documents that can be used as context for a model.

# Instructions
## Prerequisites
- **Python:** 3.10 or later https://www.python.org/downloads/
- **Package Manager:** `uv`
https://docs.astral.sh/uv/getting-started/installation/
- **Space:** Ensure you have enough disk space for the Hugging Face model weights.  
Optional: setup your HF_TOKEN from https://huggingface.co/settings/tokens

# Installation
Clone the repository and install the dependencies using the provided Makefile:
```bash
make install
```

The pipeline is driven by a CLI built with Python Fire. You must run the commands in the following order (paths are example):

1. Index corpus  
```uv run python -m src index --max_chunk_size 2000```  

2. Retrieve sources
```uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions```

3. Generation
```uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions```

# System Architecture
chunking.py: Ingests files using glob and splits them using Langchain's text splitters while preserving start/end character indices.  
indexing.py & retriever.py: Implements the search engine using bm25s. The index is persistently saved to disk.  
context.py (File Cache): Lazy-loading file cache. Instead of redundant Disk I/O operations, files are loaded into RAM once, preventing bottlenecks during dataset generation.  
generation.py: Wraps the Hugging Face transformers logic. The model is loaded only once per dataset run, optimizing VRAM and CPU usage.  
__main__.py: Exposes the CLI through python-fire, separating logic from I/O formatting.  

# Chunking Strategy
The codebase uses two distinct chunking strategies based on file extensions:
Python Code (.py): Split using Langchain's PythonCodeTextSplitter.  
Markdown/Text (.md): Split using MarkdownTextSplitter.  
For both, the chunk size is configurable via CLI (defaulting to 2000 characters to match the context length limits) with a 15% overlap to ensure context is not broken across chunks. The exact character positions (add_start_index=True) are preserved to compute the Intersection over Union (IoU) during evaluation and for the context retriever.  

# Retrieval Method
The retrieval method is BM25, implemented via the highly optimized [bm25s](https://bm25s.github.io/) library.
Unlike standard TF-IDF, BM25 handles term frequency saturation and document length normalization, making it vastly superior for code and documentation retrieval. The algorithm tokenizes the query, computes the relevance scores against the pre-built sparse matrices, and returns the top k most relevant chunks in milliseconds.

# Performance analysis
By utilizing bm25s and a RAM file cache, the retrieval throughput effortlessly processes hundreds of questions well under the 90-second constraint.  
- Expected Recall@5 (Docs): > 80% got **82%**
- Expected Recall@5 (Code): > 50% got **56%**

# Design decisions
- Decoupling Retrieval and Generation: The CLI strictly separates search and answer commands. This enforced a design where the context must be rebuilt from raw files during the answer generation phase.
- Graceful Error Handling: Degenerate inputs (empty queries, k=0, missing files) are caught gracefully returning empty sets instead of unhandled tracebacks, ensuring high robustness for automated grading.
- Pydantic Validation: All inputs/outputs between components are validated using Pydantic schemas, ensuring type safety and formatting compliance before writing the final JSONs.

# Challenges faced
- Character Index Tracking: Preserving the absolute character index of a chunk relative to the original document was challenging but resolved by utilizing Langchain's add_start_index metadata.  
- Disk I/O Bottlenecks: Initially, reading files repeatedly for context building was slow. This was solved by implementing a custom dictionary-based caching mechanism (self.file_cache).  
- Transformers Typing constraints: Dealing with strict type checkers (like Mypy/Pyright) alongside transformers dynamically generated models (like Qwen3ForCausalLM) required careful explicit typing and validation.
# Example usage
Example using the vLLM codebase.

index the corpus:
```
uv run python -m src index
```

retrieve 5 sources for a given question:
```
uv run python -m src search "How do I configure the OpenAI server?" --k 5
```

generate an answer for a question:
```
uv run python -m src answer "How do I configure the OpenAI server?" --k 5
```
The model answers: *To configure the OpenAI server, you need to set the API key and API base using the environment variables `openai_api_key` and `openai_api_base`.*

Without the context, it would have answered "I don't know."

# Resources

https://docs.astral.sh/ruff/configuration/  
https://dspy.ai/getting-started/installation/  
https://dspy.ai/diving-deeper/signatures-in-depth/  
https://docs.langchain.com/oss/python/langchain  
https://gpt.space/fr/blog/how-to-use-openai-model-temperature-for-better-ai-chat-responses  
https://huggingface.co/Qwen/Qwen3-0.6B#best-practices  

AI Usage: LLM conversational agents were used strictly as pair-programming assistants to refine architectural design choices (like the File Cache pattern), understand Mypy typing edge-cases with NumPy/Transformers, and brainstorm chunking strategies. No core logic was generated blindly without thorough understanding.  