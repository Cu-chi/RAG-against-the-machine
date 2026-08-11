import fire
from .chunking import get_files, create_documents, chunk_files
from .indexing import store_chunks, search_query


def main(max_chunk_size: int = 2000) -> None:
    extensions_files = get_files("./data/raw/",  [
        ".py",
        ".md"
    ])
    extensions_documents = create_documents(extensions_files)
    chunks = chunk_files(extensions_documents, max_chunk_size)
    store_chunks(chunks)
    results, scores = search_query("Is there any dependency?")


if __name__ == "__main__":
    fire.Fire(main)
