"""Main of the RAG project."""
import fire
import uuid
import bm25s
import sys
from dotenv import load_dotenv
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer


class RAGCLI:
    """Class used by Fire module to make the CLI."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Index the data in data/raw.

        Args:
            max_chunk_size (int, optional): maximum size of a chunk.
            Defaults to 2000.
        """
        import json
        from src.chunking import get_incremental_files, create_documents, \
            chunk_files
        from src.indexing import store_chunks
        from src.retriever import load_retriever

        if max_chunk_size <= 0 or max_chunk_size > 2000:
            print("max_chunk_size must be between 1 and 2000 characters",
                  file=sys.stderr)
            sys.exit(1)

        changed_files, current_hashes, is_incremental = get_incremental_files(
            "data/raw", [".py", ".md"], max_chunk_size)
        total_changed = sum(len(f) for f in changed_files.values())

        if is_incremental and total_changed == 0:
            print("Index is already up to date. (0 files changed)")
            return
        elif is_incremental:
            print(f"Incremental update: {total_changed}"
                  " modified/new file(s) detected")
            retriever = load_retriever()
            existing_corpus = retriever.corpus

            changed_paths = set()
            for files in changed_files.values():
                changed_paths.update(files)

            kept_chunks = []
            for doc in existing_corpus:
                if doc["metadata"]["source"] not in changed_paths:
                    from langchain_core.documents import Document
                    kept_chunks.append(Document(
                        page_content=doc["page_content"],
                        metadata=doc["metadata"]
                    ))
        else:
            kept_chunks = []

        new_documents = create_documents(changed_files)
        new_chunks = chunk_files(new_documents, max_chunk_size)

        final_chunks = kept_chunks + new_chunks
        if len(final_chunks) == 0:
            print("No documents found to index.")
            sys.exit(1)

        store_chunks(final_chunks)

        with open("data/processed/manifest.json", "w", encoding="utf-8") as f:
            json.dump({
                "max_chunk_size": max_chunk_size,
                "files": current_hashes
            }, f, indent=4)

        print(f"Ingestion complete! Indexed {len(final_chunks)} chunks "
              "under data/processed/")

    def _search_internal(self, query: str, k: int,
                         id: str = str(uuid.uuid4()),
                         retriever: bm25s.BM25 | None = None) \
            -> StudentSearchResults:
        from src.models import MinimalSearchResults, MinimalSource, \
            StudentSearchResults
        from src.retriever import search_query, load_retriever
        if k <= 0:
            print("k must be > 0", file=sys.stderr)
            sys.exit(1)
        if query == "" or query.isspace():
            print(f"Warning: Empty query for id {id}, skipping")
            search_results = MinimalSearchResults(
                question_id=id, question=query, retrieved_sources=[]
            )
            return StudentSearchResults(search_results=[search_results], k=k)
        if retriever is None:
            retriever = load_retriever()
        documents = search_query(query, retriever, k)
        minimal_sources = []
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            metadata = doc.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            minimal_sources.append(MinimalSource(
                file_path=metadata.get("source", ""),
                first_character_index=metadata.get("first_character_index", 0),
                last_character_index=metadata.get("last_character_index", 0)
            ))

        search_results = MinimalSearchResults(
            question_id=id,
            question=query,
            retrieved_sources=minimal_sources
        )

        return StudentSearchResults(
            search_results=[search_results],
            k=k
        )

    def search(self, query: str, k: int,
               id: str = str(uuid.uuid4()),
               retriever: bm25s.BM25 | None = None) -> None:
        """Search indexed documents that have a link with the query.

        Args:
            query (str): Query
            k (int): maximum number of documents
            id (str, optional): the question id.
            Defaults to str(uuid.uuid4()).
            retriever (bm25s.BM25 | None, optional): The retriever.
            Defaults to None.
        """
        from src.models import StudentSearchResults
        search_results: StudentSearchResults = self._search_internal(query, k,
                                                                     id,
                                                                     retriever)

        print(search_results.model_dump_json(indent=4))

    def search_dataset(self, dataset_path: str,
                       k: int, save_directory: str) -> None:
        """Search using a dataset.

        Args:
            dataset_path (str): path to dataset
            k (int): max number of documents to search for each question
            save_directory (str): save directory
        """
        from tqdm import tqdm
        from src.retriever import load_retriever
        from pathlib import Path
        from src.models import RagDataset, UnansweredQuestion, \
            StudentSearchResults
        if k <= 0:
            print("k must be > 0", file=sys.stderr)
            sys.exit(1)
        if not dataset_path.endswith(".json"):
            print("dataset_path must point to a .json file", file=sys.stderr)
            sys.exit(1)
        dataset = Path(dataset_path)
        if not dataset.exists():
            print(f"'{dataset_path}' doesn't exist", file=sys.stderr)
            sys.exit(1)

        json_dataset = dataset.read_text()
        rag_dataset = RagDataset.model_validate_json(json_dataset)

        search_results = []
        retriever = load_retriever()
        for question in tqdm(rag_dataset.rag_questions,
                             desc="Indexing questions"):
            if isinstance(question, UnansweredQuestion):
                results = self._search_internal(question.question, k,
                                                question.question_id,
                                                retriever)
                search_results.append(results.search_results[0])
        student_search_results = StudentSearchResults(
            search_results=search_results,
            k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_json = student_search_results.model_dump_json(indent=4)
        with open(f"{save_dir}/{dataset.name}", "w+") as f:
            f.write(model_json)

        print(f"Saved student_search_results to {save_dir}/{dataset.name}")

    def _answer_internal(self, query: str,
                         k: int) -> StudentSearchResultsAndAnswer:
        from src.retriever import load_retriever
        from src.generation import LLMGenerator
        from src.context import ContextBuilder
        from src.models import MinimalAnswer, StudentSearchResultsAndAnswer
        generator = LLMGenerator()

        retriver = load_retriever()
        search_results = self._search_internal(query, k, retriever=retriver)

        context_builder = ContextBuilder()
        context = context_builder.format_context(
            search_results.search_results[0])

        answer = generator.generate_answer(query, context)

        minimal_answer = MinimalAnswer(
            answer=answer,
            question_id=search_results.search_results[0].question_id,
            question=query,
            retrieved_sources=search_results
            .search_results[0].retrieved_sources
        )

        return StudentSearchResultsAndAnswer(
            search_results=[minimal_answer],
            k=k
        )

    def answer(self, query: str, k: int) -> None:
        """Answer to the query.

        Answer to the query with a context retrieved from
        the indexed documents.

        Args:
            query (str): query
            k (int): max number of documents for context
        """
        answer_result = self._answer_internal(query, k)

        print(answer_result.model_dump_json(indent=4))

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> None:
        """Answer each unanswered question of a dataset.

        Args:
            student_search_results_path (str): dataset with already
            searched results.
            save_directory (str): save directory
        """
        from tqdm import tqdm
        from pathlib import Path
        from src.generation import LLMGenerator
        from src.context import ContextBuilder
        from src.models import StudentSearchResults, MinimalAnswer, \
            StudentSearchResultsAndAnswer
        if not student_search_results_path.endswith(".json"):
            print("student_search_results_path must point to a .json file",
                  file=sys.stderr)
            sys.exit(1)
        results_path = Path(student_search_results_path)
        if not results_path.exists():
            print(f"'{student_search_results_path}' doesn't exist",
                  file=sys.stderr)
            sys.exit(1)

        json_results = results_path.read_text()
        search_results = StudentSearchResults.model_validate_json(json_results)
        print(f"Loaded {len(search_results.search_results)} questions")

        generator = LLMGenerator()
        context_builder = ContextBuilder()

        answers = []
        for search in tqdm(search_results.search_results,
                           desc="Generating answers"):
            context = context_builder.format_context(search)

            answer = generator.generate_answer(search.question, context)
            answers.append(MinimalAnswer(
                answer=answer,
                question_id=search.question_id,
                question=search.question,
                retrieved_sources=search.retrieved_sources
            ))

        student_search_results_and_answer = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_json = student_search_results_and_answer.model_dump_json(
            indent=4)

        with open(f"{save_dir}/{results_path.name}", "w+") as f:
            f.write(model_json)

        print("Saved student_search_results_and_answer to "
              f"{save_dir}/{results_path.name}")

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> float:
        """Debug command to evaluate the recall@k.

        It evaluates the percentage of valid source.
        A correct source counts as found when one of your results is
        in the same file and overlaps its character range.

        Args:
            student_search_results_path (str): search results path
            dataset_path (str): the dataset path

        Returns:
            float: _description_
        """
        from tqdm import tqdm
        from pathlib import Path
        from src.utils import find_question_id_index, calculate_IoU
        from src.models import RagDataset, StudentSearchResults, \
            AnsweredQuestion
        if not dataset_path.endswith(".json"):
            print("dataset_path must point to a .json file", file=sys.stderr)
            sys.exit(1)
        dataset = Path(dataset_path)
        if not dataset.exists():
            print(f"{dataset} doesn't exist", file=sys.stderr)
            sys.exit(1)

        dataset_json = dataset.read_text()
        rag_dataset = RagDataset \
            .model_validate_json(dataset_json)

        if not student_search_results_path.endswith(".json"):
            print("student_search_results_path must point to a .json file",
                  file=sys.stderr)
            sys.exit(1)
        search_results_path = Path(student_search_results_path)
        if not search_results_path.exists():
            print(f"{student_search_results_path} doesn't exist",
                  file=sys.stderr)
            sys.exit(1)

        search_results_json = search_results_path.read_text()
        search_results = StudentSearchResults \
            .model_validate_json(search_results_json)

        score = 0.0
        evaluated = 0.0
        for question in tqdm(rag_dataset.rag_questions,
                             desc="Evaluating..."):
            found = False
            results_index = find_question_id_index(question.question_id,
                                                   search_results)
            if results_index < 0:
                print(f"{question.question_id} not "
                      f"found in {search_results_path}")
                continue
            results = search_results.search_results[results_index]
            if not isinstance(question, AnsweredQuestion):
                print(f"skipping not answered question {question.question_id}")
                continue
            evaluated += 1.0
            for source in question.sources:
                for res_source in results.retrieved_sources:
                    if source.file_path == res_source.file_path:
                        if calculate_IoU(
                            source.first_character_index,
                            source.last_character_index,
                            res_source.first_character_index,
                            res_source.last_character_index,
                        ) >= 0.05:
                            found = True
            if found:
                score += 1.0
        if evaluated == 0.0:
            print("Warning: 0 question evaluated")
            return 0.0
        return score / evaluated

    def serve(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Start the local HTTP API server.

        Args:
            host (str, optional): The host. Defaults to "127.0.0.1".
            port (int, optional): The port. Defaults to 8000.
        """
        import uvicorn
        print(f"Starting server on http://{host}:{port}")
        print(f"Documentation available on http://{host}:{port}/docs")
        uvicorn.run("src.api:app", host=host, port=port)


def main() -> None:
    """Run RAG."""
    load_dotenv()
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Exited")
