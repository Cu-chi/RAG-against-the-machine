"""Module with functions used to chunk documents."""
import glob
from tqdm import tqdm
from typing import Any
from langchain_core.documents import Document
from langchain_text_splitters import PythonCodeTextSplitter, \
    MarkdownTextSplitter
import hashlib
import json
from pathlib import Path


def compute_file_hash(file_path: str) -> str:
    """Calculate the MD5 of a file.

    Args:
        file_path (str): The file

    Returns:
        str: the MD5 hash
    """
    try:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def get_incremental_files(root: str, extensions: list[str],
                          max_chunk_size: int)\
        -> tuple[dict[str, list[str]], dict[str, str], bool]:
    """Detect added or edited files using the manifest.

    Returns:
        tuple[dict[str, list[str]], dict[str, str], bool]: changed files,
        current hashes and manifest exists
    """
    all_files = get_files(root, extensions)
    current_hashes: dict[str, str] = {}
    for files in all_files.values():
        for f in files:
            current_hashes[f] = compute_file_hash(f)

    manifest_file = Path("data/processed/manifest.json")
    if not manifest_file.exists():
        return all_files, current_hashes, False

    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        old_chunk_size: int = manifest_data.get("max_chunk_size", 0)
        old_hashes: dict[str, str] = manifest_data.get("files", {})
    except Exception:
        return all_files, current_hashes, False

    if old_chunk_size != max_chunk_size:
        print(f"max_chunk_size changed ({old_chunk_size} -> {max_chunk_size})."
              " Full re-index required.")
        return all_files, current_hashes, False

    changed_files: dict[str, list[str]] = {ext: [] for ext in extensions}
    for ext, files in all_files.items():
        for f in files:
            if f not in old_hashes or old_hashes[f] != current_hashes[f]:
                changed_files[ext].append(f)

    return changed_files, current_hashes, True


def get_files(root: str, extensions: list[str]) -> dict[str, list[str]]:
    """Get all files path ending by any of extensions given.

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
    """From the list of files, return a list of document.

    Returns:
        list[Document]: the list of document
    """
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
    """Chunk documents.

    Args:
        documents (list[Document]): the documents to chunk
        max_chunk_size (int): max size of a chunk

    Returns:
        list[Document]: the documents chunked
    """
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
