import fire
import uuid
import json
from tqdm import tqdm
from .chunking import get_files, create_documents, chunk_files
from .indexing import store_chunks, search_query
from .models import MinimalSearchResults, MinimalSource, \
    RagDataset, UnansweredQuestion, StudentSearchResults, \
    AnsweredQuestion
from pathlib import Path


class RAGCLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 0:
            raise Exception
        extensions_files = get_files("data/raw/vllm-0.10.1",  [
            ".py",
            ".md"
        ])
        extensions_documents = create_documents(extensions_files)
        chunks = chunk_files(extensions_documents, max_chunk_size)
        store_chunks(chunks)

    def search(self, query: str, k: int,
               id: str = str(uuid.uuid4())) -> MinimalSearchResults:
        if k <= 0:
            raise Exception
        results, scores = search_query(query, k)

        minimal_sources = []
        for result in results[0]:
            if not isinstance(result, dict):
                continue
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            minimal_sources.append(MinimalSource(
                file_path=metadata.get("source", ""),
                first_character_index=metadata.get("first_character_index",
                                                   ""),
                last_character_index=metadata.get("last_character_index", "")
            ))

        search_results = MinimalSearchResults(
            question_id=id,
            question=query,
            retrieved_sources=minimal_sources
        )

        return search_results

    def search_dataset(self, dataset_path: str,
                       k: int, save_directory: str) -> StudentSearchResults:
        if k <= 0:
            raise Exception
        if not dataset_path.endswith(".json"):
            raise Exception
        dataset = Path(dataset_path)
        if not dataset.exists():
            raise Exception

        json_dataset = dataset.read_text()
        rag_dataset = RagDataset.model_validate_json(json_dataset)

        search_results = []
        for question in tqdm(rag_dataset.rag_questions):
            if isinstance(question, UnansweredQuestion):
                results = self.search(question.question, k,
                                      question.question_id)
                search_results.append(results)
        student_search_results = StudentSearchResults(
            search_results=search_results,
            k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(f"{save_dir}/{dataset.name}", "w+") as f:
            f.write(student_search_results.model_dump_json(indent=4))

        return student_search_results

    def answer(self, query: str, k: int) -> AnsweredQuestion:
        pass


if __name__ == "__main__":
    fire.Fire(RAGCLI)
