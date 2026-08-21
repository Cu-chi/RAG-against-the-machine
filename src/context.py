from src.models import MinimalSearchResults


class ContextBuilder:
    def __init__(self) -> None:
        self.file_cache: dict[str, str] = {}

    def get_context(self, file_path: str, start_idx: int, end_idx: int) -> str:
        if file_path not in self.file_cache:
            try:
                with open(file_path, "r", encoding="utf-8",
                          errors="ignore") as f:
                    self.file_cache[file_path] = f.read()
            except Exception as e:
                print(f"error: can't read {file_path}: {e}")
                return ""

        return self.file_cache[file_path][start_idx:end_idx]

    def format_context(self, search_results: MinimalSearchResults) -> str:
        contexts: list[str] = []
        for source in search_results.retrieved_sources:
            text = self.get_context(
                source.file_path,
                source.first_character_index,
                source.last_character_index
            )
            if text:
                contexts.append(text)

        return "\n\n---\n\n".join(contexts)
