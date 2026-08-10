import fire
from chunking import get_files, create_documents, chunk_files


def main(max_chunk_size: int = 2000) -> None:
    extensions_files = get_files("./data/raw/",  [
        ".py",
        ".md"
    ])
    for extension in extensions_files:
        print(f"{extension}: {len(extensions_files[extension])}")
    extensions_documents = create_documents(extensions_files)
    documents = chunk_files(extensions_documents, max_chunk_size)
    print(documents)


if __name__ == "__main__":
    fire.Fire(main)
