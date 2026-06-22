import fire
from chunking import get_files, create_documents, chunk_files
from storing import create_vector_store


def main() -> None:
    extensions_files = get_files("./vllm-0.10.1",  [
        ".py",
        ".md"
    ])
    for extension in extensions_files:
        print(f"{extension}: {len(extensions_files[extension])}")
    extensions_documents = create_documents(extensions_files)
    documents = chunk_files(extensions_documents)
    create_vector_store(documents, "./data/chromadb")


if __name__ == "__main__":
    fire.Fire(main)
