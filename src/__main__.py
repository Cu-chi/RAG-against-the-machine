import fire
import uuid
import bm25s
from tqdm import tqdm
from dotenv import load_dotenv
from src.chunking import get_files, create_documents, chunk_files
from src.indexing import store_chunks, search_query, load_retriever
from src.models import MinimalSearchResults, MinimalSource, \
    RagDataset, UnansweredQuestion, StudentSearchResults, \
    MinimalAnswer, StudentSearchResultsAndAnswer, AnsweredQuestion
from src.generation import LLMGenerator
from src.context import ContextBuilder
from pathlib import Path
from src.utils import find_question_id_index, calculate_IoU


class RAGCLI:
    def index(self, max_chunk_size: int = 2000) -> None:
        if max_chunk_size < 0:
            raise Exception
        extensions_files = get_files("data/raw",  [
            ".py",
            ".md"
        ])
        extensions_documents = create_documents(extensions_files)
        chunks = chunk_files(extensions_documents, max_chunk_size)
        store_chunks(chunks)

    def search(self, query: str, k: int,
               id: str = str(uuid.uuid4()),
               retriever: bm25s.BM25 | None = None) -> StudentSearchResults:
        if k <= 0:
            raise Exception
        if retriever is None:
            retriever = load_retriever()
        documents = search_query(query, retriever, k)

        minimal_sources = []
        for doc in documents[0]:
            if not isinstance(doc, dict):
                continue
            metadata = doc.get("metadata", {})
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

        return StudentSearchResults(
            search_results=[search_results],
            k=k
        )

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
        retriever = load_retriever()
        for question in tqdm(rag_dataset.rag_questions,
                             desc="Indexing questions"):
            if isinstance(question, UnansweredQuestion):
                results = self.search(question.question, k,
                                      question.question_id,
                                      retriever)
                search_results.append(results.search_results[0])
        student_search_results = StudentSearchResults(
            search_results=search_results,
            k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(f"{save_dir}/{dataset.name}", "w+") as f:
            f.write(student_search_results.model_dump_json(indent=4))

        return student_search_results

    def answer(self, query: str, k: int) -> StudentSearchResultsAndAnswer:
        generator = LLMGenerator()

        retriver = load_retriever()
        search_results = self.search(query, k, retriever=retriver)

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

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str) -> StudentSearchResultsAndAnswer:
        if not student_search_results_path.endswith(".json"):
            raise Exception
        results_path = Path(student_search_results_path)
        if not results_path.exists():
            raise Exception

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

        with open(f"{save_dir}/{results_path.name}", "w+") as f:
            f.write(
                student_search_results_and_answer.model_dump_json(indent=4))

        return student_search_results_and_answer

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> float:
        if not dataset_path.endswith(".json"):
            raise Exception
        dataset = Path(dataset_path)
        if not dataset.exists():
            raise Exception

        dataset_json = dataset.read_text()
        rag_dataset = RagDataset \
            .model_validate_json(dataset_json)

        if not student_search_results_path.endswith(".json"):
            raise Exception
        search_results_path = Path(student_search_results_path)
        if not search_results_path.exists():
            raise Exception

        search_results_json = search_results_path.read_text()
        search_results = StudentSearchResults \
            .model_validate_json(search_results_json)

        score = 0.0
        for question in rag_dataset.rag_questions:
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
        return score / len(rag_dataset.rag_questions)

    def __str__(self):
        return "a"


def main() -> None:
    load_dotenv()
    fire.Fire(RAGCLI)


if __name__ == "__main__":
    main()
