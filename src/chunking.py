import glob
from tqdm import tqdm
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import PythonCodeTextSplitter, \
    MarkdownTextSplitter


def get_files(root: str, extensions: list[str]) -> dict[str, list[str]]:
    """Get all files path ending by any of extensions given

    Args:
        extensions (list[str]): files possible extensions

    Returns:
        dict[str, list[str]]: each key is an extension with its
        associated files list
    """
    extensions_files: dict[str, list[str]] = {
        extension: [] for extension in extensions
    }
    for extension in extensions:
        extensions_files[extension] = glob.glob(f"{root}/**/*{extension}",
                                                recursive=True)
    return extensions_files


def create_documents(extensions_files: dict[str, list[str]]) \
     -> list[Document]:
    documents: list[Document] = []
    for extension, files in extensions_files.items():
        for file in files:
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    file_content: str = f.read()
                documents.append(Document(
                    page_content=file_content,
                    metadata={"source": file, "extension": extension},
                ))
            except Exception as e:
                print(f"Warning: Could not read {file}: {e}")
    return documents


def chunk_files(documents: list[Document],
                max_chunk_size: int) -> list[Document]:
    chunks: list[Document] = []
    py_splitter = PythonCodeTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=int(max_chunk_size * 0.15),
        add_start_index=True
    )
    md_splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=int(max_chunk_size * 0.15),
        add_start_index=True
    )
    for document in tqdm(documents, desc="Chunking..."):
        doc_chunks = []
        extension: str | Any = document.metadata.get("extension")

        if extension == ".py":
            doc_chunks = py_splitter.split_documents([document])
        elif extension == ".md":
            doc_chunks = md_splitter.split_documents([document])

        for chunk in doc_chunks:
            start_idx: int = chunk.metadata.get("start_index", 0)
            chunk.metadata["first_character_index"] = start_idx
            chunk.metadata["last_character_index"] = start_idx + \
                len(chunk.page_content)
            chunks.append(chunk)
    return chunks
