import glob
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
     -> dict[str, list[Document]]:
    ext_documents: dict[str, list[Document]] = {
        extension: [] for extension in extensions_files
    }
    for extension, files in extensions_files.values():
        for file in files:
            with open(file, "r", encoding="utf-8", errors="ignore") as f:
                file_content: str = f.read()
            ext_documents[extension].append(Document(
                page_content=file_content,
                metadata={"source": file, "extension": extension},
            ))
    return ext_documents


def chunk_files(ext_documents: dict[str, list[Document]]) -> list[Document]:
    chunks: list[Document] = []
    py_splitter = PythonCodeTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    md_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=300
    )
    for extension, documents in ext_documents.items():
        if extension == ".py":
            chunks.extend(py_splitter.split_documents(documents))
        elif extension == ".md":
            chunks.extend(md_splitter.split_documents(documents))
    return chunks
